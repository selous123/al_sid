import csv
import json
from tqdm import tqdm

input_csv = 'xxx/data/AL-GR-Tiny/item_info/tiny_item_sid_base.csv'
output_json = 'xxx/data/AL-GR-Tiny/item_info/tiny_sid_base_dict.json'

result = {}

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
            break
       
with open(output_json, 'w', encoding='utf-8') as json_file:
    json.dump(result, json_file, indent=2)

print(f"✅ Sucess Save File! {output_json}")
