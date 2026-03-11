"""
步骤2: 向量化 - 使用语义模型（ModelScope）
"""
import pandas as pd
import numpy as np
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer

# 从ModelScope下载模型（首次会下载，之后使用缓存）
model_dir = snapshot_download('AI-ModelScope/bge-large-zh-v1.5')

# 加载模型
print("加载模型中...")
model = SentenceTransformer(model_dir)

# 读取摘要
df = pd.read_excel('../data/conversations_with_summary.xlsx')
summaries = df['摘要'].tolist()

print(f"开始向量化 {len(summaries)} 条摘要...")

# 向量化
embeddings = model.encode(summaries, show_progress_bar=True)

# 保存
np.save('../data/embeddings.npy', embeddings)
print(f"\n向量shape: {embeddings.shape}")
print(f"示例向量前10维: {embeddings[0][:10]}")
print("向量已保存到 embeddings.npy")
