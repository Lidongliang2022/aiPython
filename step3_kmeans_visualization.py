"""
步骤3: K-Means聚类与2D可视化
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS中文支持
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
embeddings = np.load('embeddings.npy')
summaries = pd.read_csv('summaries.csv')['摘要'].tolist()

# K-Means聚类 (K=3)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(embeddings)
centroids = kmeans.cluster_centers_

# 找出每个簇的典型代表（离质心最近的样本）
representatives = []
for i in range(3):
    cluster_points = embeddings[labels == i]
    distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
    rep_idx = np.where(labels == i)[0][np.argmin(distances)]
    representatives.append((i, rep_idx, summaries[rep_idx]))
    print(f"簇{i}典型case: {summaries[rep_idx][:60]}...")

# PCA降维到2D
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)
centroids_2d = pca.transform(centroids)

# 可视化
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green']
for i in range(3):
    cluster_points = embeddings_2d[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                c=colors[i], label=f'簇{i}', alpha=0.6, s=100)

# 标记质心
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
            c='black', marker='*', s=500, label='质心', edgecolors='yellow', linewidths=2)

# 添加文本标签
for i, txt in enumerate(summaries):
    plt.annotate(f'{i}', (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                 fontsize=8, alpha=0.7)

plt.title('K-Means聚类结果 (K=3)', fontsize=16)
plt.xlabel('PCA维度1')
plt.ylabel('PCA维度2')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_clustering.png', dpi=300)
print("\n图表已保存: kmeans_clustering.png")
plt.show()
