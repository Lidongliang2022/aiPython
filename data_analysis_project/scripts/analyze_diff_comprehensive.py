"""
经纪人修改内容综合分析 - 发现AI优化方向

分析流程：
1. 统计分析：修改频率、字段分布
2. 关键词提取：发现高频添加的信息类型
3. DBSCAN聚类：发现修改模式
4. 生成优化建议
"""
import pandas as pd
import numpy as np
import json
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

# 读取数据
df = pd.read_csv('data/2000条diff结果.csv', encoding='gbk', on_bad_lines='skip')
df = df.dropna(subset=['ai_json', 'human_json'])
print(f"有效数据: {len(df)} 条\n")

# ============================================================================
# 第一步：统计分析
# ============================================================================
print("="*80)
print("第一步：统计分析 - 修改概况")
print("="*80)

# 提取修改的字段
def extract_fields(json_str):
    """提取JSON中的字段名"""
    fields = re.findall(r'(\w+):', str(json_str))
    return fields

ai_fields = []
human_fields = []
for i in range(len(df)):
    ai_fields.extend(extract_fields(df.iloc[i]['ai_json']))
    human_fields.extend(extract_fields(df.iloc[i]['human_json']))

print(f"\nAI生成最常用字段:")
for field, count in Counter(ai_fields).most_common(10):
    print(f"  {field}: {count}次")

print(f"\n经纪人修改后最常用字段:")
for field, count in Counter(human_fields).most_common(10):
    print(f"  {field}: {count}次")

# ============================================================================
# 第二步：提取修改内容并分析关键词
# ============================================================================
print("\n" + "="*80)
print("第二步：关键词分析 - 经纪人最常添加的信息")
print("="*80)

# 提取otherRemark字段的内容（这是主要修改字段）
human_remarks = []
for i in range(len(df)):
    human_json = str(df.iloc[i]['human_json'])
    match = re.search(r'otherRemark:"([^"]*)"', human_json)
    if match:
        human_remarks.append(match.group(1))

print(f"\n提取到 {len(human_remarks)} 条修改内容")

# 分词并统计高频词
all_words = []
for remark in human_remarks:
    words = jieba.lcut(remark)
    # 过滤停用词和标点
    words = [w for w in words if len(w) > 1 and w not in ['\\n', '的', '了', '和', '是', '在', '有', '个', '与', '及']]
    all_words.extend(words)

print(f"\n高频关键词（Top 30）:")
for word, count in Counter(all_words).most_common(30):
    print(f"  {word}: {count}次")

# ============================================================================
# 第三步：DBSCAN聚类分析
# ============================================================================
print("\n" + "="*80)
print("第三步：DBSCAN聚类 - 发现修改模式")
print("="*80)

if len(human_remarks) > 10:
    from sklearn.cluster import DBSCAN
    from modelscope import snapshot_download
    from sentence_transformers import SentenceTransformer

    print("\n正在加载向量化模型...")
    model_dir = snapshot_download('AI-ModelScope/bge-large-zh-v1.5')
    model = SentenceTransformer(model_dir)

    print("正在向量化修改内容...")
    embeddings = model.encode(human_remarks, show_progress_bar=True)

    print("\n正在DBSCAN聚类...")
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    labels = dbscan.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"发现 {n_clusters} 个修改模式")
    print(f"发现 {n_noise} 个异常修改（噪点）")

    # 分析每个簇
    print("\n各簇典型案例:")
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
        print(f"\n【模式 {cluster_id}】（共 {len(cluster_indices)} 条）")
        print(f"典型案例: {human_remarks[cluster_indices[0]][:100]}...")
else:
    print("\n数据量不足，跳过聚类分析")

# ============================================================================
# 第四步：生成优化建议
# ============================================================================
print("\n" + "="*80)
print("第四步：AI优化建议")
print("="*80)
print("\n基于以上分析，建议AI生成时补充以下信息：")
print("1. 查看高频关键词，这些是经纪人最常添加的信息类型")
print("2. 查看聚类结果，了解不同场景下需要补充的信息")
print("3. 重点关注：房屋特点、价格信息、配套设施、交易条件等")


