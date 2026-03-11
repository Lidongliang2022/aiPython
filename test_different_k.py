"""
测试不同K值的聚类效果
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
embeddings = np.load('embeddings.npy')
summaries = pd.read_excel('conversations_with_summary.xlsx')['摘要'].tolist()

# 测试不同的K值
k_values = [5, 10, 20]

for k in k_values:
    print(f"\n处理 K={k}...")

    # K-Means聚类
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    # 找出每个簇的典型代表
    for i in range(k):
        cluster_points = embeddings[labels == i]
        if len(cluster_points) > 0:
            distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
            rep_idx = np.where(labels == i)[0][np.argmin(distances)]
            print(f"簇{i} ({len(cluster_points)}条): {summaries[rep_idx][:40]}...")

    # PCA降维
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    centroids_2d = pca.transform(centroids)

    # 可视化
    plt.figure(figsize=(14, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, k))

    for i in range(k):
        cluster_points = embeddings_2d[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    c=[colors[i]], label=f'簇{i}', alpha=0.6, s=100)

    # 标记质心
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                c='black', marker='*', s=500, label='质心',
                edgecolors='yellow', linewidths=2)

    plt.title(f'K-Means聚类结果 (K={k})', fontsize=16)
    plt.xlabel('PCA维度1')
    plt.ylabel('PCA维度2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = f'kmeans_k{k}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"已保存: {filename}")
    plt.close()

print("\n✅ 所有K值测试完成")
