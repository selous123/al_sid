import pandas as pd
import json
import random
import os
from tqdm import tqdm

item_mapping = {}
feat_mapping = {}
cnt = 1

def process(data):
    global item_mapping, cnt  
    for row in data.itertuples():
        items = row.user_history.split(';')
        for item in items:
            if item not in item_mapping:
                item_mapping[item] = cnt
                cnt += 1
        target_items = row.target_item.split(';')
        for item in target_items:
            if item not in item_mapping:
                item_mapping[item] = cnt
                cnt += 1

def process_sequence(data_dir, save_dir):
    data = []
    global item_mapping
    df = pd.read_csv(data_dir)
    for row in df.itertuples():
        user_history = row.user_history.split(';')[::-1]
        target_item = row.target_item.split(';')
        user_history = ','.join([str(item_mapping[i]) for i in user_history])
        target_item = ','.join([str(item_mapping[i]) for i in target_item])
        data.append({'user_history': user_history,
                     'target_item': target_item})
    df = pd.DataFrame(data)
    df.to_csv(save_dir, index=False)

def process_sequence_with_feat(data_dir, save_dir):
    data = []
    global item_mapping
    global feat_mapping
    df = pd.read_csv(data_dir)
    for row in df.itertuples():
        user_history = row.user_history.split(';')[::-1]
        target_item = row.target_item.split(';')
        user_history = ','.join([str(item_mapping[i])+'|'+feat_mapping.get(i, 0) for i in user_history])
        target_item = ','.join([str(item_mapping[i]) for i in target_item])
        data.append({'user_history': user_history,
                     'target_item': target_item})
    df = pd.DataFrame(data)
    df.to_csv(save_dir, index=False)

if __name__ == '__main__':
    data_path = "xxx/data/AL-GR-Tiny"
    mapping_file = "item_mapping.json"
    mapping_path = os.path.join(data_path, mapping_file)

    feat_file = "item_info/tiny_sid_base_dict.json"
    feat_path = os.path.join(data_path, feat_file)

    print("load item mapping")
    if not os.path.isfile(mapping_path):
        data = pd.read_csv(os.path.join(data_path, 'origin_behavior/s1_tiny.csv'))
        process(data)
        data = pd.read_csv(os.path.join(data_path, 'origin_behavior/s1_tiny_test.csv'))
        process(data)

        items = list(item_mapping.items())
        random.shuffle(items) 
        item_mapping = {k: v + 1 for v, (k, _) in enumerate(items)} 
        with open(mapping_path, "w+", encoding="utf-8") as f:
            json.dump(item_mapping, f, ensure_ascii=False, indent=4)
    else:
        with open(mapping_path, "r", encoding="utf-8") as f:
            item_mapping = json.load(f)

    print("load feat mapping")
    if not os.path.isfile(feat_path):
        input_csv = os.path.join(data_path, 'item_info/tiny_item_sid_base.csv')
        output_json = os.path.join(data_path, 'item_info/tiny_sid_base_dict.json')
        with open(input_csv, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in tqdm(reader):
                item_id = row['base62_string']
                lv1 = int(row['codebook_lv1']) + 1 if row['codebook_lv1'] !='' else 0
                lv2 = int(row['codebook_lv2']) + 1 if row['codebook_lv2'] !='' else 0
                lv3 = int(row['codebook_lv3']) + 1 if row['codebook_lv3'] !='' else 0
                feat_mapping[item_id] = f"{lv1}|{lv2}|{lv3}"

        with open(output_json, 'w', encoding='utf-8') as json_file:
            json.dump(feat_mapping, json_file, indent=2)

        print(f"✅ success save file to {output_json}")
    else:
        with open(feat_path, "r", encoding="utf-8") as f:
            feat_mapping = json.load(f)

    # process_sequence(os.path.join(data_path, 'origin_behavior/s1_tiny_test.csv'), os.path.join(data_path, 'u2i/s1_tiny_test.csv'))
    # process_sequence(os.path.join(data_path, 'origin_behavior/s1_tiny.csv'), os.path.join(data_path, 'u2i/s1_tiny.csv'))

    print("start process")
    process_sequence_with_feat(os.path.join(data_path, 'origin_behavior/s1_tiny_test.csv'), os.path.join(data_path, 'u2i/s1_tiny_test_with_feat.csv'))
    process_sequence_with_feat(os.path.join(data_path, 'origin_behavior/s1_tiny.csv'), os.path.join(data_path, 'u2i/s1_tiny_with_feat.csv'))