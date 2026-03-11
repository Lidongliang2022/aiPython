"""
分析K=5聚类结果 - 找出典型场景和边缘场景
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# 加载数据
embeddings = np.load('embeddings.npy')
df = pd.read_excel('conversations_with_summary.xlsx')

# K-Means聚类 (K=5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)
centroids = kmeans.cluster_centers_

# 添加聚类标签到DataFrame
df['Cluster'] = labels

print("=" * 80)
print("K-Means (K=5) 聚类分析报告")
print("=" * 80)

# 遍历每个簇
for i in range(5):
    cluster_mask = labels == i
    cluster_embeddings = embeddings[cluster_mask]
    cluster_indices = np.where(cluster_mask)[0]

    # 计算到质心的距离
    distances = euclidean_distances(cluster_embeddings, [centroids[i]]).flatten()

    # 找出最近和最远的点
    nearest_idx = cluster_indices[np.argmin(distances)]
    farthest_idx = cluster_indices[np.argmax(distances)]

    print(f"\n{'='*80}")
    print(f"簇 {i} - 共 {len(cluster_indices)} 条数据")
    print(f"{'='*80}")

    print(f"\n【典型场景】（距离质心最近，最能代表该簇）:")
    print(f"索引: {nearest_idx}")
    print(f"距离: {distances[np.argmin(distances)]:.4f}")
    print(f"摘要: {df.iloc[nearest_idx]['摘要']}")

    print(f"\n【边缘场景】（距离质心最远，长尾/特殊情况）:")
    print(f"索引: {farthest_idx}")
    print(f"距离: {distances[np.argmax(distances)]:.4f}")
    print(f"摘要: {df.iloc[farthest_idx]['摘要']}")

# 保存带标签的DataFrame
df.to_excel('conversations_with_clusters.xlsx', index=False, engine='openpyxl')
print(f"\n{'='*80}")
print("✅ 已保存带聚类标签的数据到: conversations_with_clusters.xlsx")
print(f"{'='*80}")
