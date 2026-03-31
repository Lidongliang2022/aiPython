"""
生成带业务标注的K=3,5,10,20聚类分析报告
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# 加载数据
embeddings = np.load('data/embeddings.npy')
df = pd.read_excel('data/conversations_with_summary.xlsx')

# 定义不同K值下的业务特征（基于典型案例分析）
business_labels = {
    2: {
        0: "交易推进场景 - 价格协商、看房安排、合同签署等实质性交易推进",
        1: "咨询沟通场景 - 房源信息了解、市场行情咨询、初步意向沟通"
    },
    3: {
        0: "复杂交易场景 - 涉及税费筹划、租约处理、产权复杂等特殊问题",
        1: "常规议价场景 - 标准价格协商，买卖双方价差适中",
        2: "初步沟通场景 - 房源信息确认、市场行情了解、推广服务介绍"
    },
    4: {
        0: "强硬议价型 - 业主坚持底价，价格分歧大，协商困难",
        1: "常规议价型 - 标准价格协商，双方有谈判空间",
        2: "复杂交易型 - 涉及税费、产权、租约等复杂问题",
        3: "咨询了解型 - 初步沟通、信息确认、市场行情了解"
    },
    5: {
        0: "大价差议价场景 - 买卖双方报价差距显著，需多轮协商缩小分歧",
        1: "观望持有型 - 业主不急于出售，等待市场回暖或政策利好",
        2: "常规议价场景 - 标准价格协商流程，价差适中可谈",
        3: "咨询了解型 - 初步沟通阶段，了解房源信息与市场行情",
        4: "复杂产权税务 - 涉及经适房转商、赠与过户、税费筹划等特殊问题"
    },
    10: {
        0: "增值服务推广 - 流量聚焦、线上推广等增值服务介绍",
        1: "强硬议价型 - 业主坚持底价不让步，价格分歧明显",
        2: "高总价复杂交易 - 高价房源涉及复杂税费或产权问题",
        3: "积极促成型 - 经纪人主动推进，安排看房拍照等实质动作",
        4: "转介沟通型 - 需联系其他决策人或转交其他经纪人",
        5: "市场行情咨询 - 了解同小区成交价、市场走势等",
        6: "特殊户型议价 - 一层、顶层等特殊户型的价格协商",
        7: "产权税务咨询 - 满五唯一、个税、增值税等税务问题",
        8: "观望等待型 - 业主不急售，等待更好时机",
        9: "老旧小区改造 - 涉及装修、改造、置换等场景"
    },
    20: {
        i: f"细分场景{i}" for i in range(20)  # K=20时簇太多，先用通用标签
    }
}

def generate_report_for_k(k_value):
    print(f"\n{'='*80}")
    print(f"正在生成 K={k_value} 的分析报告...")
    print(f"{'='*80}")

    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    all_distances = euclidean_distances(embeddings, centroids)

    output = []
    output.append(f"# K-Means (K={k_value}) 聚类深度分析报告\n\n")
    output.append("## 一、整体概览\n\n")
    output.append(f"- 总样本数：{len(df)}\n")
    output.append(f"- 聚类数量：{k_value}\n")
    output.append(f"- 向量维度：{embeddings.shape[1]}\n")
    output.append(f"- Inertia（簇内误差平方和）：{kmeans.inertia_:.2f}\n\n")

    output.append("## 二、簇分布概览\n\n")
    output.append("| 簇ID | 业务场景 | 样本数 | 占比 |\n")
    output.append("|------|----------|--------|------|\n")

    for i in range(k_value):
        cluster_size = len(np.where(labels == i)[0])
        ratio = cluster_size / len(df) * 100
        biz_label = business_labels[k_value].get(i, f"场景{i}")
        output.append(f"| {i} | {biz_label} | {cluster_size} | {ratio:.1f}% |\n")

    output.append("\n")

    # 详细分析每个簇
    for i in range(k_value):
        cluster_indices = np.where(labels == i)[0]
        cluster_size = len(cluster_indices)
        biz_label = business_labels[k_value].get(i, f"场景{i}")

        output.append(f"## 三.{i+1} 簇 {i}：{biz_label}（共 {cluster_size} 条）\n\n")

        # 典型场景
        output.append(f"### 典型场景（距离质心最近的3条）\n\n")

        distances_to_centroid = all_distances[cluster_indices, i]
        nearest_3_local = np.argsort(distances_to_centroid)[:min(3, cluster_size)]
        nearest_3_global = cluster_indices[nearest_3_local]

        for rank, idx in enumerate(nearest_3_global, 1):
            dist = all_distances[idx, i]
            output.append(f"**第{rank}典型**（索引{idx}，距离质心{dist:.4f}）\n\n")

            other_dists = [all_distances[idx, j] for j in range(k_value) if j != i]
            if other_dists:
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
            if other_dists:
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
for k in [2, 3, 4, 5, 10, 20]:
    generate_report_for_k(k)

print(f"\n{'='*80}")
print("✅ 所有带业务标注的报告生成完成！")
print(f"{'='*80}")
