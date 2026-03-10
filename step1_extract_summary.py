"""
步骤1: 解析CSV并格式化对话文本
"""
import pandas as pd
import json
import re

# 直接读取原始行，提取JSON部分
conversations = []
line_count = 0

with open('30168868.csv', 'r', encoding='gbk') as f:
    next(f)  # 跳过表头

    for line in f:
        line_count += 1

        # 用正则提取JSON数组部分 [{"channel":...}]
        match = re.search(r'\[{.*?}\]', line)

        if match:
            try:
                json_str = match.group(0)
                asr_data = json.loads(json_str)

                # 格式化对话
                lines = [f"{item['role']}：{item['content']}" for item in asr_data]
                conversation = "\n".join(lines)
                conversations.append(conversation)

                print(f"[{line_count}] 已格式化")
            except Exception as e:
                print(f"[{line_count}] JSON解析失败: {e}")
                conversations.append("解析失败")
        else:
            print(f"[{line_count}] 未找到JSON")
            conversations.append("未找到JSON")

print(f"\n总共处理 {len(conversations)} 条数据")

# 保存到Excel
result_df = pd.DataFrame({
    '对话ID': range(1, len(conversations) + 1),
    '对话内容': conversations,
    '摘要': ''
})

result_df.to_excel('conversations_for_summary.xlsx', index=False, engine='openpyxl')
print(f"✅ 已保存到 conversations_for_summary.xlsx")
