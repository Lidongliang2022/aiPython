import csv

with open('30168868.csv', 'r', encoding='gbk') as f:
    reader = csv.reader(f)
    header = next(reader)
    print("Header columns:", header)
