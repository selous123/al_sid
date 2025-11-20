import os
os.environ['NCCL_DEBUG'] = 'ERROR'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from options import args
from model import SASRec
from data.dataset import DataProcess, custom_collate
from datetime import datetime
from datasets import load_dataset
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import faiss
from model import MODELS


@torch.no_grad()
def main_worker():

    print("开始加载模型...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MODELS[args.model_name](args.item_count, args, device=device)
    state_dict = torch.load(args.state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)  # 不需要 DDP 包装
    print("加载模型完成.")

    dataset = load_dataset("csv", data_files=args.data_path, split='train')
    dataset_with_tensors = dataset.map(DataProcess(max_length=args.maxlen, item_count=args.item_count, is_train=False), batched=False)
    dataloader = DataLoader(
        dataset_with_tensors,
        batch_size=50,
        num_workers=1,  # 注意不要太大，避免资源竞争
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=custom_collate
    )
    print("数据Load完成.")

    print("准备构建商品索引...")
    index = faiss.IndexFlatIP(args.hidden_units)
    index.add(model.item_emb.weight.data.cpu().numpy())
    print("商品索引构建完成...")

    Hit_20, Hit_100, Hit_500, Hit_1000 = [], [], [], []
    count = 0
    for batch in tqdm(dataloader):
        user_history, target_item = batch['user_history'], batch['target_item']
        final_feat = model.get_user_emb(user_history).cpu().numpy()
        _, indices_1000 = index.search(final_feat, 1000) #batch_size, 1000
        #print("预测结果:", indices_1000[:5,:5]) ##输出前5个样本对应的top5的商品
        for i in range(final_feat.shape[0]):
            # 当前用户的真实点击 item（非零部分）
            clicked_items = torch.masked_select(target_item[i], target_item[i] != 0).tolist()

            if not clicked_items:
                continue  # 如果没有点击记录，跳过

            hit_20, hit_100, hit_500, hit_1000 = 0.0, 0.0, 0.0, 0.0
            for item in clicked_items:
                if item in indices_1000[i]:
                    hit_1000 += 1
                if item in indices_1000[i][0:500]:
                    hit_500 += 1
                if item in indices_1000[i][0:100]:
                    hit_100 += 1
                if item in indices_1000[i][0:20]:
                    hit_20 += 1
            Hit_20.append(hit_20/len(clicked_items))
            Hit_100.append(hit_100/len(clicked_items))
            Hit_500.append(hit_500/len(clicked_items))
            Hit_1000.append(hit_1000/len(clicked_items))

    print(f"Hit Rate_20: {sum(Hit_20)/len(Hit_20):.4f}")
    print(f"Hit Rate_100: {sum(Hit_100)/len(Hit_100):.4f}")
    print(f"Hit Rate_500: {sum(Hit_500)/len(Hit_500):.4f}")
    print(f"Hit Rate_1000: {sum(Hit_1000)/len(Hit_1000):.4f}")

if __name__ == "__main__":
    main_worker()