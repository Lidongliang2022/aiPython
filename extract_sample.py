import csv
import json

try:
    with open('30168868.csv', 'r', encoding='gbk') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 5: # Get 5 samples to check length
                break
            try:
                asr = json.loads(row['asr结果'])
            except Exception as e:
                print(f"Row {i+1} JSON error: {e}")
                continue
            print(f"--- Conversation {i+1} ---")
            for item in asr:
                print(f"{item['role']}: {item['content']}")
except Exception as e:
    print(f"Error: {e}")
