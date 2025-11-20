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
from data.dataset import DataProcess, custom_collate
from datetime import datetime
from datasets import load_dataset
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
from model import MODELS
import time
from utils import count_embedding_and_dense_params


def main_worker():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    model = MODELS[args.model_name](args.item_count, args, device=device)

    # 统计参数
    emb_params, dense_params = count_embedding_and_dense_params(model)

    print("\n--- Summary ---")
    print(f"Total Embedding Parameters: {emb_params:,}")
    print(f"Total Dense (Fully Connected) Parameters: {dense_params:,}")
    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    model = model.to(device)
    print("model:", model)
    #model = DDP(model,find_unused_parameters=True)

    dataset = load_dataset("csv", data_files=args.data_path, split='train')
    data_process = DataProcess(max_length=args.maxlen, item_count=args.item_count, is_train=True, num_neg_samples=args.num_neg_samples)
    dataset_with_tensors = dataset.map(data_process, batched=False)

    #sampler = DistributedSampler(dataset_with_tensors, num_replicas=world_size, rank=rank, shuffle=False) #先不处理样本分片
    dataloader = DataLoader(
        dataset_with_tensors,
        batch_size=args.batch_size,
        num_workers=2,  # 注意不要太大，避免资源竞争
        pin_memory=True,
        prefetch_factor=2,
        sampler=None,
        collate_fn=custom_collate
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    scheduler = LinearLR(optimizer,
                     start_factor=1.0,
                     end_factor=0.0,
                     total_iters=len(dataloader) * args.epochs)
    idx = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            output = model(batch)
            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            idx += 1
            if idx % 100 == 0:
                print("Batch {}, Epoch:{}|{}, lr:{:.6f}, Loss: {:.2f}, ce_loss:{:.2f}, item_norm_loss:{:.2f}, user_norm_loss:{:.2f}".format(
                    idx, epoch, args.epochs, optimizer.param_groups[0]['lr'],
                    loss.item(),output['ce_loss'].item(),output['item_l2_loss'].item(),output['user_l2_loss'].item()),
                    flush=True)
            optimizer.step()
            scheduler.step()


        save_filename = f"{args.model_name}-{timestamp}-{epoch}.pth"
        model_dir = f"{args.model_name}_h{args.num_heads}b{args.num_blocks}"
        save_path = os.path.join(args.save_dir, model_dir, save_filename)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    

if __name__ == "__main__":
    start_time = time.time()
    main_worker()
    
    end_time = time.time()
    # 计算运行时间（秒）
    elapsed_time_seconds = end_time - start_time
    # 转换为分钟
    elapsed_time_minutes = elapsed_time_seconds / 60.0
    # 打印结果
    print(f"代码运行时间: {elapsed_time_minutes:.4f} 分钟")