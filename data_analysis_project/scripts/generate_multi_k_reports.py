"""
使用纯 NumPy 实现的 K-Means，用于生成 K=3, 10, 20 的分析报告（避开 sklearn 环境崩溃问题）
"""
import numpy as np
import pandas as pd
import os

# 设置随机种子
np.random.seed(42)

def kmeans_numpy(X, k, max_iters=100):
    # 随机初始化质心
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx]
    
    for _ in range(max_iters):
        # 计算距离
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        # 分配簇
        labels = np.argmin(distances, axis=1)
        # 更新质心
        new_centroids = np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else centroids[i] for i in range(k)])
        
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        
    return labels, centroids

def generate_report(k, X, df):
    print(f"正在生成 K={k} 的分析报告...")
    labels, centroids = kmeans_numpy(X, k)
    
    # 计算 Inertia (簇内距离平方和)
    inertia = 0
    for i in range(k):
        points = X[labels == i]
        if len(points) > 0:
            inertia += np.sum((points - centroids[i])**2)
            
    # 计算到所有质心的距离
    all_distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    
    output = []
    output.append(f"# K-Means (K={k}) 聚类评估报告\n")
    output.append("## 一、数学指标概览\n")
    output.append(f"- **Inertia (簇内误差平方和)**: {inertia:.2f} (数值越小代表类内越紧凑，但K越大值通常越小)\n")
    output.append(f"- **总样本数**: {len(df)}\n")
    output.append(f"- **聚类数量**: {k}\n\n")
    
    output.append("## 二、各簇核心场景提取\n")
    output.append("| 簇ID | 样本数 | 核心场景预览 (典型代表) |\n")
    output.append("| :--- | :--- | :--- |\n")
    
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        cluster_size = len(cluster_indices)
        
        if cluster_size > 0:
            distances_to_centroid = all_distances[cluster_indices, i]
            nearest_idx = cluster_indices[np.argmin(distances_to_centroid)]
            core_summary = df.iloc[nearest_idx]['摘要'][:50].replace('\n', ' ')
            output.append(f"| {i} | {cluster_size} | {core_summary}... |\n")
        else:
            output.append(f"| {i} | 0 | 空簇 |\n")
    
    output.append("\n---\n\n")
    
    # 详细分析
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        cluster_size = len(cluster_indices)
        if cluster_size == 0: continue
        
        distances_to_centroid = all_distances[cluster_indices, i]
        top_3_indices = cluster_indices[np.argsort(distances_to_centroid)[:3]]
        
        output.append(f"## 三、簇 {i} 详细分析（共 {cluster_size} 条数据）\n\n")
        output.append("### 典型案例（Top 3）\n\n")
        
        for rank, idx in enumerate(top_3_indices, 1):
            dist = all_distances[idx, i]
            summary = df.iloc[idx]['摘要']
            output.append(f"> **典型 {rank}** (索引 {idx}, 距离质心 {dist:.4f})\n")
            output.append(f"> 内容：{summary}\n\n")
            
        output.append("\n---\n")

    # 保存文件
    if not os.path.exists('../output'):
        os.makedirs('../output')
    report_path = f"../output/cluster_analysis_report_k{k}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(output)
    print(f"✅ 报告已成功生成: {report_path}")

if __name__ == "__main__":
    X = np.load('../data/embeddings.npy')
    df_data = pd.read_excel('../data/conversations_with_summary.xlsx')
    for val_k in [3, 10, 20]:
        generate_report(val_k, X, df_data)
