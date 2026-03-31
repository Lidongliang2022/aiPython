"""
生成 K=3,5,10,20 的完整聚类分析报告（包含典型场景和边界分析）
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# 加载数据
embeddings = np.load('data/embeddings.npy')
df = pd.read_excel('data/conversations_with_summary.xlsx')

def generate_report_for_k(k_value):
    print(f"\n{'='*80}")
    print(f"正在生成 K={k_value} 的分析报告...")
    print(f"{'='*80}")

    # K-Means聚类
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    # 计算所有距离
    all_distances = euclidean_distances(embeddings, centroids)

    # 生成报告
    output = []
    output.append(f"# K-Means (K={k_value}) 聚类深度分析报告\n\n")
    output.append("## 一、整体概览\n\n")
    output.append(f"- 总样本数：{len(df)}\n")
    output.append(f"- 聚类数量：{k_value}\n")
    output.append(f"- 向量维度：{embeddings.shape[1]}\n")
    output.append(f"- Inertia（簇内误差平方和）：{kmeans.inertia_:.2f}\n\n")

    # 簇分布概览
    output.append("## 二、簇分布概览\n\n")
    output.append("| 簇ID | 样本数 | 占比 | 核心场景预览 |\n")
    output.append("|------|--------|------|-------------|\n")

    for i in range(k_value):
        cluster_indices = np.where(labels == i)[0]
        cluster_size = len(cluster_indices)
        ratio = cluster_size / len(df) * 100

        if cluster_size > 0:
            distances_to_centroid = all_distances[cluster_indices, i]
            nearest_idx = cluster_indices[np.argmin(distances_to_centroid)]
            preview = df.iloc[nearest_idx]['摘要'][:40].replace('\n', ' ')
            output.append(f"| {i} | {cluster_size} | {ratio:.1f}% | {preview}... |\n")

    output.append("\n")

    # 详细分析每个簇
    for i in range(k_value):
        cluster_mask = labels == i
        cluster_indices = np.where(cluster_mask)[0]
        cluster_size = len(cluster_indices)

        output.append(f"## 三.{i+1} 簇 {i} 详细分析（共 {cluster_size} 条）\n\n")

        # 典型场景
        output.append(f"### 典型场景（距离质心最近的3条）\n\n")

        distances_to_centroid = all_distances[cluster_indices, i]
        nearest_3_local = np.argsort(distances_to_centroid)[:3]
        nearest_3_global = cluster_indices[nearest_3_local]

        for rank, idx in enumerate(nearest_3_global, 1):
            dist = all_distances[idx, i]
            output.append(f"**第{rank}典型**（索引{idx}，距离质心{dist:.4f}）\n\n")

            other_dists = [all_distances[idx, j] for j in range(k_value) if j != i]
            min_other_dist = min(other_dists)
            margin = min_other_dist - dist

            output.append(f"论证：到本簇质心距离{dist:.4f}，到次近簇距离{min_other_dist:.4f}，差值{margin:.4f}，属于核心代表。\n\n")
            output.append(f"内容：{df.iloc[idx]['摘要']}\n\n")

        # 边界场景
        output.append(f"### 边界场景分析\n\n")

        boundary_cases = []
        for global_idx in cluster_indices:
            dist_to_own = all_distances[global_idx, i]
            other_dists = [(j, all_distances[global_idx, j]) for j in range(k_value) if j != i]
            other_dists.sort(key=lambda x: x[1])
            closest_other_cluster, dist_to_other = other_dists[0]

            if abs(dist_to_own - dist_to_other) < 0.1:
                boundary_cases.append({
                    'idx': global_idx,
                    'dist_own': dist_to_own,
                    'other_cluster': closest_other_cluster,
                    'dist_other': dist_to_other,
                    'diff': abs(dist_to_own - dist_to_other)
                })

        if boundary_cases:
            boundary_cases.sort(key=lambda x: x['diff'])
            for rank, case in enumerate(boundary_cases[:3], 1):
                output.append(f"**边界点{rank}**（索引{case['idx']}）\n\n")
                output.append(f"论证：到簇{i}距离{case['dist_own']:.4f}，到簇{case['other_cluster']}距离{case['dist_other']:.4f}，")
                output.append(f"差值仅{case['diff']:.4f}，处于两簇交界地带。\n\n")
                output.append(f"内容：{df.iloc[case['idx']]['摘要']}\n\n")
        else:
            output.append("该簇无明显边界点。\n\n")

        output.append("---\n\n")

    # 保存报告
    report_path = f'output/cluster_analysis_report_k{k_value}.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(output)

    print(f"✅ K={k_value} 报告已生成: {report_path}")

# 生成所有K值的报告
for k in [3, 5, 10, 20]:
    generate_report_for_k(k)

print(f"\n{'='*80}")
print("✅ 所有报告生成完成！")
print(f"{'='*80}")
