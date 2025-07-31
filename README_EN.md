# Qwen3 Reranker API Service

English | [中文](README.md)

Document reranking API service based on Qwen/Qwen3-Reranker model

## Environment

### GPU Environment

**CUDA Environment**:
```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
**Other Dependencies**:
```bash
pip install -r requirements.txt
```

### Variable Configuration

```bash
# Model Configuration
# Qwen/Qwen3-Reranker-0.6B / Qwen/Qwen3-Reranker-8B
export MODEL_NAME="Qwen/Qwen3-Reranker-4B"   # Default model
export MAX_LENGTH=8192                       # Maximum sequence length

# Batch size, adjust based on GPU memory
export BATCH_SIZE=8         

# Specify GPU
export CUDA_VISIBLE_DEVICES=0  

# Model cache directory (optional)
export MODEL_CACHE_DIR="./models"           # Custom model download directory, defaults to apps/reranker/models folder

# API security configuration (optional)
export API_TOKEN="your-secret-token"

# Switch mirror source (optional)
export HF_ENDPOINT=https://hf-mirror.com      
```

## Usage

### Start Service

```bash
python api/run.py
```

The service will start at `http://localhost:23333`. See [API](API.md) documentation for interface details.

## Docker Deployment

### Build GPU Image

```bash
docker build -f Dockerfile.gpu -t qwen3-reranker-api:v0.1.0 .
```

### Run Container

```bash
# GPU version (recommended)
docker run -d -p 23333:23333 --gpus all --name qwen3-reranker \
  -v $(pwd)/api/logs:/app/api/logs \
  -v $(pwd)/apps/reranker/models:/app/apps/reranker/models \
  -e MODEL_NAME=Qwen/Qwen3-Reranker-4B \
  -e MODEL_CACHE_DIR=/app/apps/reranker/models \
  -e API_TOKEN=your-secret-token \
  -e CUDA_VISIBLE_DEVICES=0 \
  qwen3-reranker-api:v0.1.0
``` 