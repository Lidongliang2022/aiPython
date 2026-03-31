"""
方案1：基于规则的分类统计
按关键词将经纪人修改内容分类，发现优化方向
"""
import pandas as pd
import re
from collections import Counter

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

# 定义分类规则
categories = {
    '价格信息': ['价格', '万元', '底价', '报价', '成交价', '挂牌价', '调整', '议价'],
    '房屋状态': ['出租', '租客', '租户', '租约', '装修', '自住', '空置'],
    '房屋特征': ['户型', '车位', '唯一', '满五', '面积', '楼层', '朝向'],
    '产权税务': ['产权', '税费', '契税', '个税', '增值税', '过户'],
    '看房安排': ['看房', '预约', '提前', '联系', '钥匙'],
    '业主态度': ['业主', '诚心', '急售', '不急', '考虑'],
    '小区信息': ['小区', '地段', '交通', '配套', '学区'],
}

# 分类统计
category_counts = {cat: 0 for cat in categories}
category_samples = {cat: [] for cat in categories}

for remark in human_remarks:
    for category, keywords in categories.items():
        if any(kw in remark for kw in keywords):
            category_counts[category] += 1
            if len(category_samples[category]) < 3:
                category_samples[category].append(remark[:80])

# 输出结果
print("="*80)
print("经纪人修改内容分类统计")
print("="*80)
print(f"\n{'类别':<12} {'数量':<8} {'占比':<8} {'典型案例'}")
print("-"*80)

sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
for cat, count in sorted_cats:
    ratio = count / len(human_remarks) * 100
    sample = category_samples[cat][0] if category_samples[cat] else "无"
    print(f"{cat:<12} {count:<8} {ratio:>5.1f}%   {sample}...")

# 生成优化建议
print("\n" + "="*80)
print("AI优化建议")
print("="*80)
print("\n基于分类统计，AI生成时应重点补充：\n")
for i, (cat, count) in enumerate(sorted_cats[:5], 1):
    print(f"{i}. {cat}（出现{count}次）")
    print(f"   关键词：{', '.join(categories[cat][:5])}")
    print()

