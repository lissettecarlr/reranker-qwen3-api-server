"""
快速测试 Reranker 基本功能

简单快速的测试，验证模型是否能正常工作
"""

import sys
import time
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "apps"))

def quick_test():
    print("测试 Reranker...")
    
    try:
        # 导入并初始化
        from apps.reranker.reranker_predictor import RerankPredictor
        
        # 从环境变量读取模型名称
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-Reranker-4B")
        print(f"使用模型: {model_name}")
        print("加载模型中...")
        
        predictor = RerankPredictor(
            model_name=model_name,
            batch_size=2
        )
        print("模型加载成功")
        
        # 简单测试
        query = "人工智能"
        docs = [
            "人工智能是计算机科学的一个分支",
            "今天天气很好",
            "机器学习是AI的核心技术"
        ]
        
        print(f"查询: {query}")
        print("执行重排序...")
        
        start = time.time()
        result = predictor.rerank(query, docs, top_n=2)
        duration = time.time() - start
        
        print(f"重排序完成，耗时: {duration:.2f}秒")
        print("结果:")
        
        for i, item in enumerate(result['results'], 1):
            score = item['relevance_score']
            doc_idx = item['index']
            doc_text = result['documents'][i-1]
            print(f"  {i}. [{score:.3f}] {doc_text}")
        
        print("测试通过！Reranker 功能正常")
        return True
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n测试结束")
    else:
        print("\n请检查环境配置和依赖安装")
    
    sys.exit(0 if success else 1) 