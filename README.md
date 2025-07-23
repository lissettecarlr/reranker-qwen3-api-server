# Qwen3 Reranker API 服务

基于 Qwen/Qwen3-Reranker 模型的文档重排序 API 服务

## 环境

### GPU 环境

**CUDA 环境**:
```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
**其他依赖**:
```bash
pip install -r requirements.txt
```

### 变量配置

```bash
# 模型配置
# Qwen/Qwen3-Reranker-0.6B / Qwen/Qwen3-Reranker-8B
export MODEL_NAME="Qwen/Qwen3-Reranker-4B"   # 默认模型
export MAX_LENGTH=8192                       # 最大序列长度

# 批处理大小，根据现存大小调整
export BATCH_SIZE=8         

# 指定GPU
export CUDA_VISIBLE_DEVICES=0  

# 模型缓存目录 (可选)
export MODEL_CACHE_DIR="./models"           # 自定义模型下载目录，默认为apps/reranker/models文件夹

# API安全配置 (可选)
export API_TOKEN="your-secret-token"

# 切换镜像源（可选）
export HF_ENDPOINT=https://hf-mirror.com      
```

## 使用

### 启动服务

```bash
python api/run.py
```

服务将在 `http://localhost:23333` 启动。接口说明见文档[API](API.md)


## Docker 部署

### 构建 GPU 镜像

```bash
docker build -f Dockerfile.gpu -t qwen3-reranker-api:v0.1.0 .
```

### 运行容器

```bash
# GPU 版本 (推荐)
docker run -d -p 23333:23333 --gpus all --name qwen3-reranker \
  -v $(pwd)/api/logs:/app/api/logs \
  -v $(pwd)/apps/reranker/models:/app/apps/reranker/models \
  -e MODEL_NAME=Qwen/Qwen3-Reranker-4B \
  -e MODEL_CACHE_DIR=/app/apps/reranker/models \
  -e API_TOKEN=your-secret-token \
  -e CUDA_VISIBLE_DEVICES=0 \
  qwen3-reranker-api:v0.1.0
```







