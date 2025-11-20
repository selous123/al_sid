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


# 初始化分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size):
    setup(rank, world_size)
    is_main = dist.get_rank() == 0

    if is_main and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    model = MODELS[args.model_name](args.item_count, args, device=rank)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    torch.cuda.set_device(rank)  # local_rank 是当前进程分配的 GPU 编号
    model = model.to(rank)
    model = DDP(model,find_unused_parameters=True)

    dataset = load_dataset("csv", data_files=args.data_path, split='train')
    data_process = DataProcess(max_length=args.maxlen, item_count=args.item_count, is_train=True, num_neg_samples=args.num_neg_samples)
    dataset_with_tensors = dataset.map(data_process, batched=False)

    sampler = DistributedSampler(dataset_with_tensors, num_replicas=world_size, rank=rank, shuffle=False) #先不处理样本分片
    dataloader = DataLoader(
        dataset_with_tensors,
        batch_size=args.batch_size,
        num_workers=2,  # 注意不要太大，避免资源竞争
        pin_memory=True,
        prefetch_factor=2,
        sampler=sampler,
        collate_fn=custom_collate
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    scheduler = LinearLR(optimizer,
                     start_factor=1.0,
                     end_factor=0.0,
                     total_iters=len(dataloader) * args.epochs)

    idx = 0
    # for batch in tqdm(
    #         dataloader,
    #         disable=not is_main  # 只有主进程显示进度条
    #     ):
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

        if rank == 0:
            dist.barrier()  # 所有进程都会等待到这一步再继续执行
            save_filename = f"{args.model_name}-{timestamp}-{epoch}.pth"
            save_path = os.path.join(args.save_dir, args.model_name, save_filename)
            torch.save(model.module.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        else:
            dist.barrier()  # 其他进程也要等待，确保不会提前退出

    cleanup()


if __name__ == "__main__":
    # world_size = torch.cuda.device_count()
    world_size = 1
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)