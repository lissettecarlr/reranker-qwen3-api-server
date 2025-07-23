"""
Reranker应用连接器 - API直接调用这里的函数
"""
import sys
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "apps"))

# 导入Reranker相关
from apps.reranker.reranker_predictor import RerankPredictorManager, format_rerank_results
import torch

# 配置日志
logger = logging.getLogger(__name__)

# 安全配置
MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", "100"))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "4096"))
MAX_DOCUMENT_LENGTH = int(os.getenv("MAX_DOCUMENT_LENGTH", "8192"))

# 全局预测器
predictor_manager = RerankPredictorManager.get_instance()
current_predictor = None

# 启动时预加载模型
def _preload_model():
    """启动时预加载默认模型"""
    global current_predictor
    
    default_model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-Reranker-4B")
    cache_dir = os.getenv("MODEL_CACHE_DIR")  # 可选的自定义缓存目录
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        current_predictor = predictor_manager.init_predictor(
            model_name=default_model_name,
            device=device,
            max_length=int(os.getenv("MAX_LENGTH", "8192")),
            batch_size=int(os.getenv("BATCH_SIZE", "8")),
            cache_dir=cache_dir
        )
        cache_info = f", 缓存目录: {cache_dir}" if cache_dir else ", 缓存目录: 默认(apps/reranker/models)"
        logger.info(f"Reranker模型预加载成功: {default_model_name}, 设备: {device}{cache_info}")
    except Exception as e:
        logger.error(f"Reranker模型预加载失败: {str(e)}")
        current_predictor = None

# 执行预加载
_preload_model()

def process_request(input_data: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """处理API请求 - API直接调用此函数"""
    global current_predictor
    
    if config is None:
        config = {}
    
    # 模型配置
    model_name = config.get('model_name', os.getenv("MODEL_NAME", "Qwen/Qwen3-Reranker-4B"))
    
    try:
        # 检查是否需要重新加载模型
        default_model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-Reranker-4B")
        cache_dir = os.getenv("MODEL_CACHE_DIR")  # 获取缓存目录配置
        
        if model_name != default_model_name:
            # 使用非默认模型，需要重新加载
            device = "cuda" if torch.cuda.is_available() else "cpu"
            current_predictor = predictor_manager.init_predictor(
                model_name=model_name,
                device=device,
                max_length=config.get('max_length', 8192),
                batch_size=config.get('batch_size', 8),
                cache_dir=cache_dir
            )
        elif not current_predictor:
            # 默认模型但预加载失败，返回错误
            return _create_error_response("Reranker模型未加载，请检查模型配置")
        
        # 验证输入数据格式
        if not isinstance(input_data, dict):
            return _create_error_response("输入格式错误，必须是字典格式")
        
        # 提取必需字段
        query = input_data.get('query')
        documents = input_data.get('documents')
        
        if not query:
            return _create_error_response("缺少必需字段: query")
        
        if not documents:
            return _create_error_response("缺少必需字段: documents")
        
        # 验证字段类型和长度
        if not isinstance(query, str):
            return _create_error_response("query 必须是字符串")
        
        if not isinstance(documents, list):
            return _create_error_response("documents 必须是列表")
        
        # 验证长度限制
        if len(query.strip()) == 0:
            return _create_error_response("query 不能为空")
        
        if len(query) > MAX_QUERY_LENGTH:
            return _create_error_response(f"query 长度超过限制，最大 {MAX_QUERY_LENGTH} 字符")
        
        if len(documents) == 0:
            return _create_error_response("documents 不能为空")
        
        if len(documents) > MAX_DOCUMENTS:
            return _create_error_response(f"documents 数量超过限制，最多 {MAX_DOCUMENTS} 个")
        
        # 验证每个文档
        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                return _create_error_response(f"第 {i+1} 个文档必须是字符串")
            
            if len(doc.strip()) == 0:
                return _create_error_response(f"第 {i+1} 个文档不能为空")
            
            if len(doc) > MAX_DOCUMENT_LENGTH:
                return _create_error_response(f"第 {i+1} 个文档长度超过限制，最大 {MAX_DOCUMENT_LENGTH} 字符")
        
        # 清理数据
        query = query.strip()
        documents = [doc.strip() for doc in documents]
        
        # 执行重排序
        start_time = time.time()
        top_n = input_data.get('top_n')
        instruction = config.get('instruction')
        
        # 验证 top_n
        if top_n is not None:
            if not isinstance(top_n, int) or top_n <= 0:
                return _create_error_response("top_n 必须是正整数")
            if top_n > len(documents):
                return _create_error_response("top_n 不能大于文档数量")
        
        results = current_predictor.rerank(
            query=query,
            documents=documents,
            top_n=top_n,
            instruction=instruction
        )
        
        processing_time = time.time() - start_time
        
        # 格式化响应
        response = {
            "status": "success",
            "results": results["results"],
            "metadata": {
                **results["metadata"],
                "total_processing_time_ms": round(processing_time * 1000, 2)
            }
        }
        
        return response
        
    except ValueError as e:
        # 输入验证错误，可以返回给用户
        logger.warning(f"输入验证失败: {str(e)}")
        return _create_error_response(str(e))
    except Exception as e:
        # 内部错误，不暴露详细信息
        logger.error(f"处理rerank请求时发生内部错误: {str(e)}", exc_info=True)
        return _create_error_response("服务暂时不可用，请稍后重试")

def get_app_info():
    """获取应用信息 - API调用"""
    return {
        'name': "RerankApp",
        'version': "1.0.0",
        'status': 'ready',
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'model_loaded': current_predictor is not None,
        'model_info': current_predictor.get_model_info() if current_predictor else None
    }

def _create_error_response(message: str) -> Dict[str, Any]:
    """创建统一的错误响应"""
    return {
        'error': message,
        'status': 'failed'
    }

# 启动时打印当前应用信息
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-Reranker-4B")

print("="*50)
print("应用: RerankApp v1.0.0")
print(f"模型: {model_name}")
print(f"设备: {device}")
print("="*50) 