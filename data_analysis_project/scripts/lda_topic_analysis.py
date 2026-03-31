"""
方案2：LDA主题模型分析
发现经纪人修改内容的隐含主题
"""
import pandas as pd
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 读取数据
df = pd.read_csv('data/2000条diff结果.csv', encoding='gbk', on_bad_lines='skip')
df = df.dropna(subset=['ai_json', 'human_json'])

# 提取人工备注
human_remarks = []
for i in range(len(df)):
    match = re.search(r'otherRemark:"([^"]*)"', str(df.iloc[i]['human_json']))
    if match:
        human_remarks.append(match.group(1))

print(f"有效数据: {len(human_remarks)} 条\n")

# 分词
stop_words = ['\\n', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', '的', '了', '和', '是', '在', '有', '个', '与', '及']
docs = []
for remark in human_remarks:
    words = jieba.lcut(remark)
    words = [w for w in words if len(w) > 1 and w not in stop_words]
    docs.append(' '.join(words))

# LDA主题建模
vectorizer = CountVectorizer(max_features=100)
X = vectorizer.fit_transform(docs)

n_topics = 5
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)

# 输出主题
feature_names = vectorizer.get_feature_names_out()
print("="*80)
print("LDA主题分析结果")
print("="*80)

for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"\n主题 {topic_idx+1}: {', '.join(top_words)}")
