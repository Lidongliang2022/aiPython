"""
步骤4: DBSCAN挖掘未知Bad Case
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 模拟20条Bad Case向量（围绕3个中心+噪点）
np.random.seed(42)
center1 = np.random.randn(6, 768) * 0.3 + [1, 0, 0] * 256
center2 = np.random.randn(7, 768) * 0.3 + [-1, 1, 0] * 256
center3 = np.random.randn(5, 768) * 0.3 + [0, -1, 1] * 256
noise = np.random.randn(2, 768) * 2  # 孤立噪点

bad_cases = np.vstack([center1, center2, center3, noise])

# DBSCAN聚类
dbscan = DBSCAN(eps=15, min_samples=3)
labels = dbscan.fit_predict(bad_cases)

# 统计结果
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"发现 {n_clusters} 个错误簇")
print(f"发现 {n_noise} 个孤立噪点")
print(f"聚类标签: {labels}")

# PCA降维可视化
pca = PCA(n_components=2)
bad_cases_2d = pca.fit_transform(bad_cases)

plt.figure(figsize=(10, 8))
unique_labels = set(labels)
colors = ['red', 'blue', 'green', 'orange', 'purple']

for k in unique_labels:
    if k == -1:
        col = 'gray'
        marker = 'x'
        label = '噪点'
    else:
        col = colors[k % len(colors)]
        marker = 'o'
        label = f'簇{k}'

    class_member_mask = (labels == k)
    xy = bad_cases_2d[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=col, marker=marker,
                s=150, label=label, alpha=0.7, edgecolors='black')

plt.title('DBSCAN Bad Case挖掘', fontsize=16)
plt.xlabel('PCA维度1')
plt.ylabel('PCA维度2')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../output/dbscan_bad_cases.png', dpi=300)
print("\n图表已保存: ../output/dbscan_bad_cases.png")
plt.show()
