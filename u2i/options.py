import os
import time
import torch
import argparse

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='xxx/data/AL-GR-Tiny/u2i/s1_tiny.csv', type=str)
parser.add_argument('--num_neg_samples', default=20, type=int)
parser.add_argument('--item_count', default=24573855, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--lr', default=5e-3, type=float)
parser.add_argument('--maxlen', default=100, type=int)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.05, type=float)
parser.add_argument('--state_dict_path', default="./ckpt", type=str)
# parser.add_argument('--norm_first', action='store_true', default=False)
parser.add_argument('--save_dir', default='./ckpt', type=str)
parser.add_argument('--model_name', default='basemodel', type=str)


args = parser.parse_args()