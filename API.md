# Qwen3 Reranker API 文档

本文档详细介绍 Qwen3 Reranker 文档重排序服务的所有 API 接口。

## 基础信息

- **服务地址**: `http://localhost:23333`
- **API版本**: v1
- **Content-Type**: `application/json`
- **兼容性**: Cohere Rerank API 格式兼容

## 认证授权

当服务端设置了 API Token 时，所有请求都需要在请求头中包含 Authorization 字段：

```bash
Authorization: Bearer your-secret-token
```

## 接口总览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/rerank` | 文档重排序 |
| GET  | `/api/v1/rerank/model-info` | 获取模型信息 |
| GET  | `/api/v1/rerank/health` | 健康检查 |

## 限制和约束

- **查询长度**: 最大 4096 字符
- **文档数量**: 最多 100 个文档
- **文档长度**: 每个文档最大 8192 字符
- **并发请求**: 建议控制在合理范围内以避免 GPU 内存溢出

---

## 1. 文档重排序

### 接口信息
- **路径**: `POST /api/v1/rerank`
- **功能**: 根据查询对文档列表进行相关性重排序
- **格式**: Cohere API 兼容

### 请求参数

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| query | string | 是 | 搜索查询，最大 4096 字符 |
| documents | array | 是 | 文档列表，最多 100 个 |
| top_n | integer | 否 | 返回前 N 个结果，默认返回所有 |
| model | string | 否 | 模型标识符，用于 API 兼容性 |
| instruction | string | 否 | 自定义指令，覆盖默认相关性判断 |

### 基础使用示例

```bash
curl -X POST "http://localhost:23333/api/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是机器学习？",
    "documents": [
      "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。",
      "深度学习是机器学习的子集，使用神经网络处理复杂数据。",
      "Python是一种流行的编程语言，广泛用于数据科学和机器学习。",
      "数据库是存储和管理数据的系统。",
      "云计算提供按需的计算资源和服务。"
    ]
  }'
```

### 带参数的示例

```bash
curl -X POST "http://localhost:23333/api/v1/rerank" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token" \
  -d '{
    "query": "深度学习算法原理",
    "documents": [
      "卷积神经网络是深度学习中用于图像处理的核心算法。",
      "循环神经网络适合处理序列数据，如文本和时间序列。",
      "Transformer架构彻底改变了自然语言处理领域。",
      "Python是编程语言的一种。",
      "数据预处理是机器学习项目的重要步骤。"
    ],
    "top_n": 3,
    "model": "Qwen/Qwen3-Reranker-4B",
    "instruction": "根据深度学习算法的相关性对文档进行排序"
  }'
```

### 响应格式

```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.8956
    },
    {
      "index": 1,
      "relevance_score": 0.7342
    },
    {
      "index": 2,
      "relevance_score": 0.6789
    }
  ],
  "metadata": {
    "query": "什么是机器学习？",
    "total_documents": 5,
    "returned_documents": 3,
    "processing_time_ms": 1250.5,
    "model": "Qwen/Qwen3-Reranker-4B",
    "device": "cuda",
    "api_processing_time_ms": 1251.2
  }
}
```


---

## 2. 获取模型信息

### 接口信息
- **路径**: `GET /api/v1/rerank/model-info`
- **功能**: 获取当前加载的模型详细信息

### 请求示例

```bash
curl -H "Authorization: Bearer your-secret-token" \
  "http://localhost:23333/api/v1/rerank/model-info"
```

### 响应格式

```json
{
  "model_name": "Qwen/Qwen3-Reranker-4B",
  "device": "cuda",
  "is_loaded": true,
  "cuda_available": true,
  "app_info": {
    "name": "RerankApp",
    "version": "1.0.0",
    "status": "ready",
    "device": "cuda",
    "model_loaded": true,
    "model_info": {
      "model_name": "Qwen/Qwen3-Reranker-4B",
      "device": "cuda",
      "dtype": "torch.float16",
      "max_length": 8192,
      "batch_size": 8,
      "is_loaded": true,
      "cuda_available": true
    }
  }
}
```

---

## 3. 健康检查

### 接口信息
- **路径**: `GET /api/v1/rerank/health`
- **功能**: 检查服务状态和模型加载情况

### 请求示例

```bash
curl "http://localhost:23333/api/v1/rerank/health"
```

### 响应格式

**服务正常**:
```json
{
  "status": "healthy",
  "service": "rerank-api",
  "version": "1.0.0",
  "model_loaded": true,
  "app_status": "ready",
  "timestamp": 1704067200.123
}
```


---

## 数据格式说明

### 相关性分数 (relevance_score)
- **范围**: 0.0 - 1.0
- **含义**: 文档与查询的相关性分数，数值越高表示越相关
- **计算**: 基于 Qwen3-Reranker 模型的概率输出

### 文档索引 (index)
- **含义**: 文档在原始输入列表中的位置（从 0 开始）
- **用途**: 用于追溯排序后的文档对应原始输入的哪个文档

### 元数据字段说明

| 字段 | 类型 | 描述 |
|------|------|------|
| query | string | 原始查询 |
| total_documents | integer | 输入文档总数 |
| returned_documents | integer | 返回文档数量 |
| processing_time_ms | float | 模型推理耗时（毫秒） |
| model | string | 使用的模型名称 |
| device | string | 运行设备（cuda/cpu） |
| api_processing_time_ms | float | API 总耗时（毫秒） |

---

