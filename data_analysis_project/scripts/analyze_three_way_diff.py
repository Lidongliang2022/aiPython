"""
三方对比分析：通话原文 vs AI提取 vs 经纪人修改

目标：区分两类问题
1. AI遗漏：通话中提到了，但AI没提取
2. 经纪人补充：通话中没提到，经纪人凭经验添加
"""
import pandas as pd
import re
from collections import Counter

# 读取数据
df = pd.read_csv('/Users/lidongliang021/Downloads/2000条diff结果（含原文）.csv',
                 encoding='gbk', on_bad_lines='skip')
df = df.dropna()
print(f"有效数据: {len(df)} 条\n")

# 查看样本
print("="*80)
print("数据样本")
print("="*80)
for i in range(min(2, len(df))):
    print(f"\n【样本 {i+1}】")
    print(f"通话原文: {str(df.iloc[i]['通话原文(ASR)'])[:150]}...")
    print(f"AI提取: {str(df.iloc[i]['AI提取结果_ai_json'])[:150]}...")
    print(f"人工修改: {str(df.iloc[i]['人工最终结果_human_json'])[:150]}...")
