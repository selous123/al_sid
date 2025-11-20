import csv
import json
from tqdm import tqdm

# 输入和输出文件路径
input_csv = '/home/admin/workspace/aop_lab/data/AL-GR-Tiny/item_info/tiny_item_sid_base.csv'
output_json = '/home/admin/workspace/aop_lab/data/AL-GR-Tiny/item_info/tiny_sid_base_dict.json'

result = {}

# 读取 CSV 并构建字典
with open(input_csv, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in tqdm(reader):
        item_id = row['base62_string']
        lv1 = int(row['codebook_lv1']) + 1 if row['codebook_lv1'] !='' else 0
        lv2 = int(row['codebook_lv2']) + 1 if row['codebook_lv2'] !='' else 0
        lv3 = int(row['codebook_lv3']) + 1 if row['codebook_lv3'] !='' else 0
        result[item_id] = f"{lv1}|{lv2}|{lv3}"
        if result[item_id] == '||':
            print(row)
            print("结果为空") #RfIgP
            break
       
# 保存为 JSON 文件
with open(output_json, 'w', encoding='utf-8') as json_file:
    json.dump(result, json_file, indent=2)

print(f"✅ 已成功将数据保存到 {output_json}")
