"""
分析经纪人修改diff数据，提取优化方向
"""
import pandas as pd
import json
import re
from collections import Counter

# 读取数据
print("正在读取数据...")
try:
    df = pd.read_csv('data/2000条diff结果.csv', encoding='gbk', on_bad_lines='skip')
    print(f"成功读取 {len(df)} 条数据")
except Exception as e:
    print(f"读取失败: {e}")
    exit(1)

# 清理数据
df = df.dropna(subset=['ai_json', 'human_json'])
print(f"清理后剩余 {len(df)} 条有效数据")

# 查看前5个样本
print("\n" + "="*80)
print("前5个样本预览")
print("="*80)
for i in range(min(5, len(df))):
    print(f"\n样本 {i+1}:")
    print(f"AI生成: {str(df.iloc[i]['ai_json'])[:200]}")
    print(f"人工修改: {str(df.iloc[i]['human_json'])[:200]}")
