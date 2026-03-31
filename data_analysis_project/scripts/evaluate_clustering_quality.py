"""
计算不同K值的轮廓系数（Silhouette Score）
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 加载数据
embeddings = np.load('data/embeddings.npy')

print("=" * 80)
print("聚类质量评估指标对比")
print("=" * 80)
print(f"\n{'K值':<6} {'Inertia':<10} {'轮廓系数':<12} {'DB指数':<12} {'CH指数':<12}")
print("-" * 80)

results = []
for k in [2, 3, 4, 5, 10]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(embeddings, labels)
    db_score = davies_bouldin_score(embeddings, labels)
    ch_score = calinski_harabasz_score(embeddings, labels)

    results.append({
        'k': k,
        'inertia': inertia,
        'silhouette': silhouette,
        'db': db_score,
        'ch': ch_score
    })

    print(f"{k:<6} {inertia:<10.2f} {silhouette:<12.4f} {db_score:<12.4f} {ch_score:<12.2f}")

print("\n" + "=" * 80)
print("指标说明：")
print("- 轮廓系数（Silhouette）：[-1, 1]，越接近1越好，表示簇内紧凑、簇间分离")
print("- DB指数（Davies-Bouldin）：越小越好，表示簇间分离度高")
print("- CH指数（Calinski-Harabasz）：越大越好，表示簇间方差与簇内方差比值高")
print("=" * 80)

# 找出最优K值
best_silhouette = max(results, key=lambda x: x['silhouette'])
best_db = min(results, key=lambda x: x['db'])
best_ch = max(results, key=lambda x: x['ch'])

print(f"\n最优K值（按不同指标）：")
print(f"- 轮廓系数最高：K={best_silhouette['k']} (score={best_silhouette['silhouette']:.4f})")
print(f"- DB指数最低：K={best_db['k']} (score={best_db['db']:.4f})")
print(f"- CH指数最高：K={best_ch['k']} (score={best_ch['ch']:.2f})")
