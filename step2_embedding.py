"""
步骤2: 本地向量化 (Embedding)
"""
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# 加载中文向量模型（首次运行会自动下载）
model = SentenceTransformer('shibing624/text2vec-base-chinese')

# 读取摘要数据
df = pd.read_csv('summaries.csv')
summaries = df['摘要'].tolist()

print(f"开始向量化 {len(summaries)} 条摘要...")

# 转换为向量（768维）
embeddings = model.encode(summaries, show_progress_bar=True)

# 保存向量
np.save('embeddings.npy', embeddings)
print(f"\n向量shape: {embeddings.shape}")
print(f"示例向量前10维: {embeddings[0][:10]}")
print("向量已保存到 embeddings.npy")
