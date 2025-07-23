"""
Qwen3 Reranker 预测器封装

基于 Qwen/Qwen3-Reranker 模型的文档重排序预测器
支持文档列表根据查询相关性进行重新排序
"""

import os
import sys
import time
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import threading

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class RerankPredictor:
    """
    Qwen3 Reranker 预测器
    
    支持多种模型：
    - Qwen/Qwen3-Reranker-4B (默认)

    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-4B",
        device: str = None,
        max_length: int = 8192,
        batch_size: int = 8,
        cache_dir: str = None,
        **kwargs
    ):
        """
        初始化 Reranker 预测器
        
        Parameters
        ----------
        model_name : str
            HuggingFace 模型名称或本地路径
        device : str
            计算设备，默认自动选择
        max_length : int
            最大序列长度
        batch_size : int
            批处理大小
        cache_dir : str
            模型缓存目录，默认使用项目下的 models 文件夹
        """
        self.model_name = model_name
        if device is None:
            if torch.cuda.is_available():
                # 在多GPU环境中默认使用第一个GPU
                self.device = "cuda:0" if torch.cuda.device_count() > 1 else "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.max_length = max_length
        self.batch_size = batch_size
        
        # 设置模型缓存目录
        if cache_dir is None:
            # 使用 reranker 应用目录下的 models 文件夹
            reranker_dir = Path(__file__).parent  # apps/reranker/
            self.cache_dir = reranker_dir / "models"
        else:
            self.cache_dir = Path(cache_dir)
        
        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型和分词器
        self.model = None
        self.tokenizer = None
        
        # 预处理常量
        self.PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        
        # Token IDs
        self.token_false_id = None
        self.token_true_id = None
        self.prefix_tokens = None
        self.suffix_tokens = None
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载模型和分词器"""
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                logger.info(f"正在加载 Reranker 模型: {self.model_name} (尝试 {retry_count + 1}/{max_retries + 1})")
                start_time = time.time()
                
                # 第一次失败后，清理缓存重试
                if retry_count > 0:
                    logger.warning("检测到加载失败，清理缓存重试...")
                    self._clear_cache()
                
                # 添加额外的参数来避免 tokenizer 问题
                tokenizer_kwargs = {
                    "padding_side": "left",
                    "cache_dir": self.cache_dir,
                    "trust_remote_code": True,  # 信任远程代码
                    "use_fast": False,  # 使用慢速 tokenizer 避免 rust 问题
                    "force_download": retry_count > 0,  # 重试时强制重新下载
                }
                
                # 尝试加载 tokenizer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name, 
                        **tokenizer_kwargs
                    )
                except Exception as tokenizer_error:
                    # 如果 tokenizer 失败，尝试强制重新下载
                    error_str = str(tokenizer_error)
                    logger.warning(f"Tokenizer 加载失败: {tokenizer_error}")
                    
                    if "Consistency check failed" in error_str or "size" in error_str:
                        logger.info("检测到文件损坏，强制重新下载...")
                        tokenizer_kwargs["force_download"] = True
                    else:
                        logger.info("尝试使用慢速 tokenizer...")
                        tokenizer_kwargs["use_fast"] = False
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        **tokenizer_kwargs
                    )
                
                # 加载模型
                model_kwargs = {
                    "torch_dtype": self.dtype,
                    "cache_dir": self.cache_dir,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,  # 减少 CPU 内存使用
                    "force_download": retry_count > 0,  # 重试时强制重新下载
                }
                
                if self.device.startswith("cuda"):
                    # 检查GPU数量，避免多GPU设备分配问题
                    if torch.cuda.device_count() > 1:
                        # 多GPU环境，使用单个GPU避免设备不一致
                        model_kwargs["device_map"] = {"": 0}  # 强制使用GPU 0
                        logger.info(f"检测到多GPU环境，强制使用 cuda:0")
                    else:
                        model_kwargs["device_map"] = "auto"
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                ).to(self.device).eval()
                
                # 预处理 token IDs
                self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
                self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
                self.prefix_tokens = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
                self.suffix_tokens = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)
                
                load_time = time.time() - start_time
                logger.info(f"模型加载完成，耗时: {load_time:.2f}s，设备: {self.device}")
                
                # 加载成功，跳出重试循环
                break
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                
                if "data did not match any variant of untagged enum ModelWrapper" in error_msg:
                    logger.error("检测到 tokenizer.json 文件损坏，正在清理缓存...")
                elif "OutOfMemoryError" in error_msg:
                    logger.error("GPU 内存不足，请考虑使用 CPU 或更小的模型")
                elif "Connection" in error_msg or "timeout" in error_msg.lower():
                    logger.error("网络连接问题，请检查网络或使用镜像源")
                
                if retry_count <= max_retries:
                    logger.warning(f"模型加载失败 (尝试 {retry_count}/{max_retries + 1}): {error_msg}")
                    logger.info(f"等待 2 秒后重试...")
                    time.sleep(2)
                else:
                    logger.error(f"模型加载最终失败: {error_msg}")
                    
                    # 提供详细的错误信息和解决建议
                    suggestions = []
                    if "data did not match any variant of untagged enum ModelWrapper" in error_msg:
                        suggestions.extend([
                            "1. tokenizer.json 文件损坏，已自动清理缓存",
                            "2. 尝试更新 transformers: pip install -U transformers",
                            "3. 检查网络连接，或设置镜像: export HF_ENDPOINT=https://hf-mirror.com"
                        ])
                    elif "OutOfMemoryError" in error_msg:
                        suggestions.extend([
                            "1. 使用 CPU 模式: device='cpu'",
                            "2. 减少 batch_size",
                            "3. 使用更小的模型: Qwen/Qwen3-Reranker-1B"
                        ])
                    else:
                        suggestions.extend([
                            "1. 检查网络连接",
                            "2. 确保模型名称正确",
                            "3. 检查 transformers 版本兼容性"
                        ])
                    
                    suggestion_text = "\n".join(suggestions)
                    raise RuntimeError(
                        f"Failed to load reranker model after {max_retries + 1} attempts: {error_msg}\n\n"
                        f"建议解决方案:\n{suggestion_text}"
                    )
    
    def _clear_cache(self):
        """清理模型缓存"""
        try:
            import shutil
            model_cache_path = self.cache_dir / f"models--{self.model_name.replace('/', '--')}"
            if model_cache_path.exists():
                logger.info(f"清理缓存目录: {model_cache_path}")
                shutil.rmtree(model_cache_path)
            
            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"清理缓存失败: {str(e)}")
    
    def _format_instruction(
        self, 
        query: str, 
        doc: str, 
        instruction: Optional[str] = None
    ) -> str:
        """格式化指令"""
        if instruction is None:
            instruction = "Evaluate how relevant the following document is to the query for retrieving useful information to answer or provide context for the query"
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
    
    def _process_inputs(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
        """处理输入对"""
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        
        for i in range(len(inputs["input_ids"])):
            inputs["input_ids"][i] = self.prefix_tokens + inputs["input_ids"][i] + self.suffix_tokens
        
        inputs = self.tokenizer.pad(
            inputs, 
            padding=True, 
            return_tensors="pt", 
            max_length=self.max_length
        )
        
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        return inputs
    
    @torch.no_grad()
    def _compute_scores(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        """计算相关性分数"""
        try:
            logits = self.model(**inputs).logits[:, -1, :]
            true_scores = logits[:, self.token_true_id]
            false_scores = logits[:, self.token_false_id]
            
            scores = torch.stack([false_scores, true_scores], dim=1)
            scores = torch.nn.functional.log_softmax(scores, dim=1)
            
            # 返回 "yes" 的概率
            result = scores[:, 1].exp().cpu().tolist()
            
            # 清理 GPU 张量
            del logits, true_scores, false_scores, scores
            
            return result
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"CUDA 内存不足，建议减少 batch_size。当前: {self.batch_size}"
            ) from e
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        对文档进行重新排序
        
        Parameters
        ----------
        query : str
            查询文本
        documents : List[str]
            待排序的文档列表
        top_n : Optional[int]
            返回前 N 个结果，默认返回所有
        instruction : Optional[str]
            自定义指令
            
        Returns
        -------
        Dict[str, Any]
            包含排序结果的字典
        """
        if not documents:
            return {"results": []}
        
        start_time = time.time()
        
        # 创建查询-文档对
        pairs = [
            self._format_instruction(query, doc, instruction) 
            for doc in documents
        ]
        
        # 批量处理
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            inputs = self._process_inputs(batch_pairs)
            scores = self._compute_scores(inputs)
            all_scores.extend(scores)
            
            # 清理 CUDA 缓存
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        
        # 创建结果列表
        results = [
            {"index": i, "relevance_score": score, "document": doc}
            for i, (score, doc) in enumerate(zip(all_scores, documents))
        ]
        
        # 按相关性分数降序排序
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # 应用 top_n 限制
        if top_n is not None and top_n > 0:
            results = results[:top_n]
        
        processing_time = time.time() - start_time
        
        return {
            "results": [
                {"index": r["index"], "relevance_score": r["relevance_score"]}
                for r in results
            ],
            "documents": [r["document"] for r in results],
            "metadata": {
                "query": query,
                "total_documents": len(documents),
                "returned_documents": len(results),
                "processing_time_ms": round(processing_time * 1000, 2),
                "model": self.model_name,
                "device": self.device
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "cache_dir": str(self.cache_dir),
            "is_loaded": self.model is not None,
            "cuda_available": torch.cuda.is_available()
        }


class RerankPredictorManager:
    """Reranker 预测器管理器 - 单例模式"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.predictors = {}  # 存储不同配置的预测器
        self.current_predictor = None
        
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def init_predictor(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-4B",
        device: str = None,
        max_length: int = 8192,
        batch_size: int = 8,
        cache_dir: str = None,
        **kwargs
    ) -> RerankPredictor:
        """
        初始化或获取预测器
        
        使用配置作为键，避免重复加载相同模型
        """
        # 创建配置键，包含缓存目录信息
        cache_dir_key = str(cache_dir) if cache_dir else "default"
        config_key = f"{model_name}_{device}_{max_length}_{batch_size}_{cache_dir_key}"
        
        if config_key not in self.predictors:
            logger.info(f"创建新的 Reranker 预测器: {config_key}")
            self.predictors[config_key] = RerankPredictor(
                model_name=model_name,
                device=device,
                max_length=max_length,
                batch_size=batch_size,
                cache_dir=cache_dir,
                **kwargs
            )
        
        self.current_predictor = self.predictors[config_key]
        return self.current_predictor
    
    def get_current_predictor(self) -> Optional[RerankPredictor]:
        """获取当前预测器"""
        return self.current_predictor
    
    def list_predictors(self) -> List[str]:
        """列出已加载的预测器"""
        return list(self.predictors.keys())
    
    def clear_predictors(self):
        """清理所有预测器"""
        self.predictors.clear()
        self.current_predictor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 全局实例
predictor_manager = RerankPredictorManager()


def format_rerank_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """格式化 rerank 结果为标准 API 响应格式"""
    return {
        "status": "success",
        "results": results["results"],
        "metadata": results["metadata"]
    } 