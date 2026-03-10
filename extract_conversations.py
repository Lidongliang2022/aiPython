"""
抽取并格式化对话文本
将ASR结果转换为统一格式：经纪人/业主对话
"""
import pandas as pd
import json

# 读取CSV
df = pd.read_csv('30168868.csv', encoding='gbk')
print(f"加载 {len(df)} 条数据")

# 格式化对话
formatted_conversations = []

for idx, row in df.iterrows():
    asr_data = json.loads(row['asr结果'])

    # 拼接对话，统一格式
    lines = []
    for item in asr_data:
        role = item['role']
        content = item['content']
        lines.append(f"{role}：{content}")

    conversation_text = "\n".join(lines)
    formatted_conversations.append(conversation_text)

    print(f"[{idx+1}/{len(df)}] 已处理")

# 保存到Excel
result_df = pd.DataFrame({
    '对话ID': range(1, len(formatted_conversations) + 1),
    '对话内容': formatted_conversations
})

result_df.to_excel('formatted_conversations.xlsx', index=False, engine='openpyxl')
print(f"\n✅ 已保存 {len(formatted_conversations)} 条格式化对话到 formatted_conversations.xlsx")
