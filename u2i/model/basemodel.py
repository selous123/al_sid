import torch
import torch.nn as nn
from torch.nn import functional as F

class BaseModel(torch.nn.Module):
    def __init__(self, item_num, args, device):
        super(BaseModel, self).__init__()
        self.item_num = item_num
        self.args = args
        self.dev = device

        self.user_l2_lambda = 0.05
        self.item_l2_lambda = 1.0

        ##定义emb
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.position_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        # === 新增：两层MLP用于用户表征增强 ===
        # 可以设置中间维度，这里假设和 hidden_units 一致，也可改为其他值

        self.user_mlp = nn.Sequential(
            nn.Linear(args.hidden_units, args.hidden_units*2),
            nn.ReLU(),
            nn.Linear(args.hidden_units*2, args.hidden_units*4),
            nn.ReLU(),
            nn.Linear(args.hidden_units*4, args.hidden_units)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


    def get_user_emb(self, user_seq):
        # print("i am in basemodel")
        user_seq_tensor = user_seq[:,:,0].to(self.dev)  # [B, seq_len, 1]->[B, seq_len] ##对应的item_id
        #user_seq_tensor = torch.LongTensor(user_seq).to(self.dev)  # [B, seq_len, 1]->[B, seq_len] ##对应的item_id
        user_seq_emb = self.item_emb(user_seq_tensor)  # [B, seq_len, dim]
        # 创建 mask: 非 padding 的位置为 1
        seq_mask = torch.not_equal(user_seq_tensor, 0).int()  # [B, seq_len]

        batch_size, seq_len = user_seq_tensor.shape

        # 生成 position indices: [1, 2, ..., seq_len]，复制到 batch 维度
        poss = torch.arange(1, seq_len + 1, device=self.dev).repeat(batch_size, 1)  # [B, seq_len]

        # 使用 mask 屏蔽 padding 位置的位置索引（虽然不 strictly 必要，但更干净）
        poss = poss * seq_mask  # padding 位置变为 0，其余保持原 position index

        # 获取 position embedding 并加到 item embedding 上
        poss_emb = self.position_emb(poss)  # [B, seq_len, dim]，padding 位置使用 padding_idx=0

        # 将 item embedding 和 position embedding 相加
        seq_emb = user_seq_emb + poss_emb  # [B, seq_len, dim]
        
        # Compute masked mean
        user_emb = (seq_emb * seq_mask.unsqueeze(-1)).sum(dim=1)  # [B, dim]
        valid_length = seq_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        user_emb = user_emb / valid_length  # [B, dim]

        user_emb = self.user_mlp(user_emb)
        return user_emb


    def get_item_emb(self, pos_seq, neg_seq):
        pos_emb = self.item_emb(pos_seq.squeeze(1).to(self.dev)) #[bs, dim]
        neg_emb = self.item_emb(neg_seq.view(-1).to(self.dev)) #[bs * num_neg, dim]
        # print("shape of pos_embs:", pos_emb.shape)
        # print("shape of neg_embs:", neg_emb.shape)
        return pos_emb, neg_emb


    def get_loss(self, user_emb, pos_emb, neg_emb):
        #计算损失函数
        # user_emb: [bs, dim]
        # pos_emb: [bs, dim]
        # neg_emb: [bs * num_neg, dim]
        pos_score =  (user_emb * pos_emb).sum(dim=1, keepdim=True)  # [bs, 1]
        neg_score = (user_emb.unsqueeze(1) * neg_emb.unsqueeze(0)).sum(dim=-1)  # [bs, total_neg,dim]->[bs, total_neg]

        logits = torch.cat([pos_score, neg_score], dim=1)  # [bs, 1 + total_neg]
        # Step 4: Softmax over all candidates; label is the first one (index 0)
        labels = torch.zeros(user_emb.shape[0], dtype=torch.long, device=logits.device)  # all positive at index 0
        ce_loss = F.cross_entropy(logits, labels, reduction='mean')

        #增加l2_norm的损失
        # ========== 2. 用户嵌入 L2 正则 ==========
        user_norm_loss = user_emb.pow(2).sum(1).mean(0)  # 所有 user_emb 的 L2 范数平方和
        # ========== 3. 物品嵌入 L2 正则（包括正样本和负样本）==========
        # item_norm_loss = pos_emb.pow(2).sum(1).mean(0)
        item_norm_loss = (pos_emb.pow(2).sum(1).mean() + neg_emb.pow(2).sum(dim=1).mean()) / 2

        # ========== 4. 总损失 ==========
        loss = ce_loss + \
            self.user_l2_lambda * user_norm_loss + \
            self.item_l2_lambda * item_norm_loss
        #loss = ce_loss

        # 可选：返回各部分 loss 用于监控
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "user_l2_loss": user_norm_loss,
            "item_l2_loss": item_norm_loss
        }


    def forward(self, batch):
        user_seq, pos_seq, neg_seq = batch['user_history'], batch['target_item'], batch['negative_items']
        #准备数据
        user_emb = self.get_user_emb(user_seq)
        pos_emb, neg_emb = self.get_item_emb(pos_seq, neg_seq)

        # 获取包含正则项的损失（dict 形式）
        loss_dict = self.get_loss(
            user_emb, pos_emb, neg_emb
        )

        total_loss = loss_dict["loss"]

        return {
            "loss": total_loss,
            "ce_loss": loss_dict["ce_loss"],
            "user_l2_loss": loss_dict["user_l2_loss"],
            "item_l2_loss": loss_dict["item_l2_loss"],
            "user_emb": user_emb,
            "pos_emb": pos_emb,
            "neg_emb": neg_emb
        }