import numpy as np
import random
from datasets import load_dataset
from torch.utils.data import DataLoader

import torch
from torch.nn.utils.rnn import pad_sequence
# from options import args

class DataProcess:
    def __init__(self, max_length, item_count, is_train=True, num_neg_samples=1):
        self.max_length = max_length
        self.is_train = is_train
        self.item_count = item_count
        self.num_neg_samples = num_neg_samples  # 每个样本采多少个负样本
        self.input_column = 'user_history'
        self.output_column = 'target_item'

    def __call__(self, example):
        #user_history = [int(i) for i in example[self.input_column].split(',')] #支持将user_history从一维转成二维list
        input_data = example[self.input_column]
        user_history = [list(map(int, item.split('|'))) for item in input_data.split(',')]
        if self.output_column in example:
            target_item = [int(i) for i in example[self.output_column].split(',')] if not self.is_train else [int(example[self.output_column])]

        # 填充历史记录到 max_length
        if len(user_history) < self.max_length:
            #user_history = user_history + [0] * (self.max_length - len(user_history))
            pad_item = [0] * len(user_history[0]) if user_history else [0]  # 动态确定维度，避免空情况
            user_history = user_history + [pad_item] * (self.max_length - len(user_history))
        else:
            user_history = user_history[-self.max_length:]

        # 采样负样本
        if self.is_train:
            negative_items = random.sample(range(1, self.item_count + 1), k=self.num_neg_samples)
            return {
                "user_history": user_history,
                "target_item": target_item,
                "negative_items": negative_items
            }
        else:
            return {
                "user_history": user_history,
                "target_item": target_item
            }


def custom_collate(batch):
    # Step 1: user_history 是 List[List[List[int]]] → 转为 (B, L, C) tensor
    user_history = torch.tensor([item["user_history"] for item in batch], dtype=torch.long)  # [B, L, C]

    # Step 2: target_item 可能长度不同，需要 pad
    target_item = [torch.tensor(item["target_item"], dtype=torch.long) for item in batch]
    target_item = pad_sequence(target_item, batch_first=True, padding_value=0)  # [B, T_max]

    # Step 3: negative_items（训练时存在）
    if "negative_items" in batch[0]:
        negative_items = torch.tensor([item["negative_items"] for item in batch], dtype=torch.long)  # [B, K]
        return {
            "user_history": user_history,
            "target_item": target_item,
            "negative_items": negative_items
        }
    else:
        return {
            "user_history": user_history,
            "target_item": target_item
        }




if __name__ == '__main__':
    dataset = load_dataset("csv", data_files="/home/admin/workspace/aop_lab/data/AL-GR-Tiny/u2i/s1_tiny.csv", split='train', streaming=True)

    #padding在右边
    dataset_with_tensors = dataset.map(DataProcess(max_length=100, item_count=24573855, is_train=True, num_neg_samples=5), batched=False)
    torch_dataset = dataset_with_tensors
    dataloader = DataLoader(
        torch_dataset,
        batch_size=2,
        num_workers=1,  # 注意不要太大，避免资源竞争
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=custom_collate
    )

    for batch in dataloader:
        print(batch)
        break
    # import pandas as pd
    # data = pd.read_csv('data/AL-GR/u2i/s1_tiny.csv')
    # data[0:10000].to_csv('data/AL-GR/u2i/s1_test.csv')