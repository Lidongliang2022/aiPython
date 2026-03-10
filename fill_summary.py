"""
自动填充摘要 - 使用DeepSeek API（免费）
"""
import pandas as pd
from openai import OpenAI

# DeepSeek API配置（请替换为你的API Key）
client = OpenAI(
    api_key="sk-xxx",  # 在 https://platform.deepseek.com 获取
    base_url="https://api.deepseek.com"
)

# 读取Excel
df = pd.read_excel('conversations_for_summary.xlsx')
print(f"加载 {len(df)} 条对话")

# 逐条生成摘要
for idx, row in df.iterrows():
    conversation = row['对话内容']

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"请用200字左右总结这段房产通话的核心信息（客户意向、价格讨论、房况、约看情况、冲突点等）：\n\n{conversation}"
            }],
            temperature=0
        )

        summary = response.choices[0].message.content
        df.at[idx, '摘要'] = summary
        print(f"[{idx+1}/{len(df)}] {summary[:50]}...")

    except Exception as e:
        print(f"[{idx+1}] 失败: {e}")
        df.at[idx, '摘要'] = "生成失败"

# 保存
df.to_excel('conversations_with_summary.xlsx', index=False, engine='openpyxl')
print(f"\n✅ 已保存到 conversations_with_summary.xlsx")
