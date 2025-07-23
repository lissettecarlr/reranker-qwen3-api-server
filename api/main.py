from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager

import time
import logging
import os

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 导入应用连接器（自动初始化）
    try:
        import config.apps
        logger.info("Reranker应用连接器加载完成")
    except Exception as e:
        logger.error(f"应用连接器加载失败: {str(e)}")
    
    yield
    
    # 关闭时清理
    logger.info("Reranker服务关闭中...")

app = FastAPI(title="Qwen3 Reranker API服务", version="1.0.0", lifespan=lifespan)

# 添加CORS中间件 - 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有源，生产环境应改为具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

# Authorization中间件
class AuthorizationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, token: str = None):
        super().__init__(app)
        self.token = token
        self.enabled = token is not None and token.strip() != ""
        if self.enabled:
            logger.info("Authorization校验已启用")
        else:
            logger.info("Authorization校验未启用")
    
    async def dispatch(self, request: Request, call_next):
        # 如果未启用校验，直接通过
        if not self.enabled:
            return await call_next(request)
        
        # 跳过根路径和文档路径的校验
        if request.url.path in ["/", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)
        
        # 检查Authorization头
        authorization = request.headers.get("Authorization")
        if not authorization:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "缺少Authorization头"}
            )
        
        # 验证Bearer token格式
        try:
            scheme, token = authorization.split(" ", 1)
            if scheme.lower() != "bearer":
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Authorization头格式错误，应为: Bearer <token>"}
                )
        except ValueError:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Authorization头格式错误，应为: Bearer <token>"}
            )
        
        # 验证token
        if token != self.token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "无效的token"}
            )
        
        return await call_next(request)

# 从环境变量读取token
API_TOKEN = os.getenv("API_TOKEN")

# 添加Authorization中间件
app.add_middleware(AuthorizationMiddleware, token=API_TOKEN)

# 时间中间件
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"请求 {request.url.path} 耗时: {process_time:.2f}秒")
        return response

app.add_middleware(TimingMiddleware)

# 引入API路由
from v1.rerank_api import router as rerank_router

app.include_router(rerank_router, prefix="/api/v1", tags=["rerank-api"])

@app.get("/")
async def root():
    """Reranker API服务根路径"""
    # 导入应用连接器获取信息
    import config.apps as app_connector
    return {
        "message": "Qwen3 Reranker API服务",
        "version": "1.0.0",
        "service": "rerank",
        "app_info": app_connector.get_app_info(),
        "api_docs": "/docs",
        "endpoints": {
            "rerank": "/api/v1/rerank",
            "model_info": "/api/v1/rerank/model-info",
            "health": "/api/v1/rerank/health"
        }
    }