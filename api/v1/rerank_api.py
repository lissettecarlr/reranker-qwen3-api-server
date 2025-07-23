"""
同步 Reranking API 接口

提供文档重排序的同步 API 接口，兼容 Cohere Rerank API 格式
支持直接调用，无需异步任务队列
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import logging
import time
import sys
from pathlib import Path

# 添加路径
sys.path.extend([
    str(Path(__file__).parent),
    str(Path(__file__).parent.parent),
    str(Path(__file__).parent.parent.parent)
])

# 直接导入应用连接器
import config.apps as app_connector

logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter()

# 环境变量配置
MAX_DOCUMENTS = 100
MAX_QUERY_LENGTH = 4096
MAX_DOCUMENT_LENGTH = 8192

class RerankRequest(BaseModel):
    """Reranking 请求模型 - Cohere API 兼容格式"""
    model: Optional[str] = Field(
        None, 
        description="模型标识符（为了API兼容性）"
    )
    query: str = Field(
        ..., 
        max_length=MAX_QUERY_LENGTH,
        description="搜索查询"
    )
    documents: List[str] = Field(
        ..., 
        max_length=MAX_DOCUMENTS,
        description="待重排序的文档列表"
    )
    top_n: Optional[int] = Field(
        None, 
        ge=1, 
        le=MAX_DOCUMENTS,
        description="返回前N个结果的数量"
    )
    instruction: Optional[str] = Field(
        None,
        description="自定义指令（可选）"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("查询不能为空")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f"查询长度超过限制，最大 {MAX_QUERY_LENGTH} 字符")
        return v.strip()

    @field_validator("documents")
    @classmethod
    def validate_documents(cls, v):
        if not v:
            raise ValueError("文档列表不能为空")
        if len(v) > MAX_DOCUMENTS:
            raise ValueError(f"文档数量超过限制，最多 {MAX_DOCUMENTS} 个文档")

        for i, doc in enumerate(v):
            if not doc or not doc.strip():
                raise ValueError(f"第 {i+1} 个文档不能为空")
            if len(doc) > MAX_DOCUMENT_LENGTH:
                raise ValueError(f"第 {i+1} 个文档长度超过限制，最大 {MAX_DOCUMENT_LENGTH} 字符")

        return [doc.strip() for doc in v]

    @field_validator("top_n")
    @classmethod
    def validate_top_n(cls, v, info):
        if v is not None:
            documents = info.data.get("documents", [])
            if v > len(documents):
                raise ValueError("top_n 不能大于文档数量")
        return v


class RerankResult(BaseModel):
    """单个重排序结果"""
    index: int = Field(description="文档在原始列表中的索引")
    relevance_score: float = Field(description="相关性分数")


class RerankResponse(BaseModel):
    """重排序响应模型 - Cohere API 兼容格式"""
    results: List[RerankResult] = Field(description="重排序结果列表")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据信息")


class ModelInfoResponse(BaseModel):
    """模型信息响应"""
    model_config = {"protected_namespaces": ()}
    
    model_name: str
    device: str
    is_loaded: bool
    cuda_available: bool
    app_info: Dict[str, Any]


@router.post("/rerank", response_model=RerankResponse, summary="文档重排序")
async def rerank_documents(request: RerankRequest):
    """
    对文档进行重排序
    
    根据查询和文档的相关性进行重新排序，返回相关性分数最高的文档。
    API 格式兼容 Cohere Rerank API。
    
    参数:
    - **query**: 搜索查询字符串
    - **documents**: 待重排序的文档列表
    - **top_n**: 可选，返回前N个结果的数量
    - **model**: 可选，模型标识符（为了API兼容性）
    - **instruction**: 可选，自定义指令
    
    返回:
    - **results**: 重排序结果列表，包含索引和相关性分数
    - **metadata**: 处理元数据信息
    """
    try:
        start_time = time.time()
        
        logger.info(f"收到重排序请求 - 查询长度: {len(request.query)}, 文档数量: {len(request.documents)}, top_n: {request.top_n}")
        
        # 构建输入数据
        input_data = {
            "query": request.query,
            "documents": request.documents,
            "top_n": request.top_n
        }
        
        # 构建配置
        config = {}
        if request.model:
            config["model_name"] = request.model
        if request.instruction:
            config["instruction"] = request.instruction
        
        # 调用应用连接器处理请求
        result = app_connector.process_request(input_data, config)
        
        # 检查处理结果
        if result.get("status") == "failed":
            logger.warning(f"重排序处理失败: {result.get('error', '未知错误')}")
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "处理失败")
            )
        
        processing_time = time.time() - start_time
        
        # 构建响应
        response_data = {
            "results": result["results"],
            "metadata": {
                **result.get("metadata", {}),
                "api_processing_time_ms": round(processing_time * 1000, 2)
            }
        }
        
        logger.info(f"重排序完成 - 处理时间: {processing_time:.3f}s, 返回结果数量: {len(result['results'])}")
        
        return RerankResponse(**response_data)
        
    except HTTPException:
        # 重新抛出 HTTP 异常
        raise
    except ValueError as e:
        # 输入验证错误
        logger.warning(f"输入验证失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 内部服务器错误
        logger.error(f"重排序 API 内部错误: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="服务暂时不可用，请稍后重试"
        )


@router.get("/rerank/model-info", response_model=ModelInfoResponse, summary="获取模型信息")
async def get_model_info():
    """
    获取当前加载的模型信息
    
    返回模型名称、设备信息、加载状态等详细信息
    """
    try:
        app_info = app_connector.get_app_info()
        model_info = app_info.get("model_info", {})
        
        response_data = {
            "model_name": model_info.get("model_name", "未知"),
            "device": model_info.get("device", "未知"),
            "is_loaded": app_info.get("model_loaded", False),
            "cuda_available": model_info.get("cuda_available", False),
            "app_info": app_info
        }
        
        return ModelInfoResponse(**response_data)
        
    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="无法获取模型信息"
        )


@router.get("/rerank/health", summary="健康检查")
async def health_check():
    """
    API 健康检查
    
    检查服务状态和模型加载情况
    """
    try:
        app_info = app_connector.get_app_info()
        
        return {
            "status": "healthy" if app_info.get("model_loaded") else "degraded",
            "service": "rerank-api",
            "version": "1.0.0",
            "model_loaded": app_info.get("model_loaded", False),
            "app_status": app_info.get("status", "unknown"),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "service": "rerank-api",
            "version": "1.0.0",
            "error": "服务异常",
            "timestamp": time.time()
        } 