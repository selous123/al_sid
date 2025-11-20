import numpy as np
import torch
from options import args
from datasets import load_dataset
from torch.utils.data import DataLoader
from .basemodel import BaseModel
from .modules import PointWiseFeedForward  # 假设已定义
# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(BaseModel):
    def __init__(self, item_num, args, device):
        super(SASRec, self).__init__(item_num, args, device)

        self.user_l2_lambda = 0.05
        self.item_l2_lambda = 0.5
        #创建sasrec对应的网络层
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
        
        self.apply(self._init_weights)


    def get_user_emb(self, user_seq):
        # print("i am in sasrec")
        user_seq_tensor = user_seq[:,:,0].to(self.dev)  # [B, seq_len]
        seq_mask = user_seq_tensor != 0  # boolean mask [B, seq_len]
        batch_size, seq_len = user_seq_tensor.shape

        # seq Embedding
        user_seq_emb = self.item_emb(user_seq_tensor)  # [B, seq_len, dim]
        #user_seq_emb *= self.item_emb.embedding_dim ** 0.5  # Scale (optional, from Transformer)

        # Position Embedding
        positions = torch.arange(1, seq_len + 1, device=self.dev).unsqueeze(0).repeat(batch_size, 1)  # [B, seq_len]
        pos_emb = self.position_emb(positions)  # [B, seq_len, dim]
        seq_emb = user_seq_emb + pos_emb  # [B, seq_len, dim]

        # Embedding dropout
        seq_emb = self.emb_dropout(seq_emb)

        # === Attention Mask: causal (prevent attending to future) ===
        # We want: lower triangular = True (can attend), upper = False (masked out)
        # torch.tril: includes diagonal
        # Upper triangle = True → mask out
        # tensor([[False,  True,  True,  True],
                # [False, False,  True,  True],
                # [False, False, False,  True],
                # [False, False, False, False]])
        attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=self.dev), diagonal=1).bool()

        # Now seq_emb: [B, seq_len, dim], but MHA expects [seq_len, B, dim]
        seq_emb = seq_emb.transpose(0, 1)  # → [seq_len, batch_size, dim]

        # === Transformer Blocks ===
        for i in range(len(self.attention_layers)):
            # if self.norm_first:
            # Norm -> MHA -> Add -> Norm -> FFN -> Add
            norm_x = self.attention_layernorms[i](seq_emb)
            mha_outputs, _ = self.attention_layers[i](
                norm_x, norm_x, norm_x,
                attn_mask=attention_mask,      # future masking
                need_weights=False
            )
            seq_emb = seq_emb + mha_outputs

            # Forward layer
            norm_y = self.forward_layernorms[i](seq_emb)
            ffn_outputs = self.forward_layers[i](norm_y)
            seq_emb = seq_emb + ffn_outputs
            # else:
            #     # MHA -> Add -> Norm -> FFN -> Add -> Norm
            #     mha_outputs, _ = self.attention_layers[i](
            #         seq_emb, seq_emb, seq_emb,
            #         attn_mask=attention_mask,
            #         need_weights=False
            #     )
            #     seq_emb = self.attention_layernorms[i](seq_emb + mha_outputs)

            #     ffn_outputs = self.forward_layers[i](seq_emb)
            #     seq_emb = self.forward_layernorms[i](seq_emb + ffn_outputs)

        # After all blocks: [seq_len, batch_size, dim] -> [batch_size, seq_len, dim]
        seq_emb = seq_emb.transpose(0, 1)  # [B, seq_len, dim]
        log_feats = self.last_layernorm(seq_emb)  # Final layer norm

        # 获取最后一个有效 item 的位置
        # seq_mask: [B, seq_len], 记录非 padding 位置
        last_valid_idx = seq_mask.sum(dim=1) - 1  # 每个序列最后一个非零元素索引
        user_emb = log_feats[torch.arange(batch_size), last_valid_idx, :]  # [B, dim]

        return user_emb

