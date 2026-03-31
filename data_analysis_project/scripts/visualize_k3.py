"""
生成K=3的K-Means聚类可视化图
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
embeddings = np.load('data/embeddings.npy')
summaries = pd.read_excel('data/conversations_with_summary.xlsx')['摘要'].tolist()

# K-Means聚类 (K=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)
centroids = kmeans.cluster_centers_

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

plt.title('K-Means聚类结果 (K=3)', fontsize=16)
plt.xlabel('PCA维度1')
plt.ylabel('PCA维度2')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('output/kmeans_k3.png', dpi=300)
print("✅ K=3可视化图已保存: output/kmeans_k3.png")
plt.show()
