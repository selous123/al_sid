from typing import Tuple, Union
from torch import nn
import torch
from .model_utils import VisualTransformer
from .configuration_bert import BertConfig
from .modeling_bert import BertModel
import numpy as np
from utils import _tokenizer

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 vocab_size: int,
                 text_attention_probs_dropout_prob: float,
                 text_hidden_act: str,
                 text_hidden_dropout_prob: float,
                 text_hidden_size: int,
                 text_initializer_range: float,
                 text_intermediate_size: int,
                 text_max_position_embeddings: int,
                 text_num_attention_heads: int,
                 text_num_hidden_layers: int,
                 text_type_vocab_size: int,
                 tokenizer=_tokenizer,
                 ):
        super().__init__()

        vision_heads = vision_width // 64
        self.visual = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.bert_config = BertConfig(
            vocab_size_or_config_json_file=vocab_size,
            hidden_size=text_hidden_size,
            num_hidden_layers=text_num_hidden_layers,
            num_attention_heads=text_num_attention_heads,
            intermediate_size=text_intermediate_size,
            hidden_act=text_hidden_act,
            hidden_dropout_prob=text_hidden_dropout_prob,
            attention_probs_dropout_prob=text_attention_probs_dropout_prob,
            max_position_embeddings=text_max_position_embeddings,
            type_vocab_size=text_type_vocab_size,
            initializer_range=text_initializer_range,
            layer_norm_eps=1e-12,
        )
        self.bert = BertModel(self.bert_config)

        self.text_projection = nn.Parameter(
            torch.empty(text_hidden_size, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.tokenizer = tokenizer

        # loss
        # self.cl_head = CLIPLoss_withMask_withmultimodal()

        self.initialize_parameters()

    def initialize_parameters(self):
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection,
                            std=self.bert_config.hidden_size ** -0.5)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        pad_index = self.tokenizer.vocab['[PAD]']
        attn_mask = text.ne(pad_index).type(self.dtype)
        x = self.bert(text, attention_mask=attn_mask)[0].type(
            self.dtype)  # [batch_size, seq_length, hidden_size]
        x = x @ self.text_projection
        return x[:, 0, :], x

    def forward(self, image, text):
        assert image is not None or text is not None, "text and image cannot both be None!"

        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        features = {'image_embed': image_features,
                    'text_embed': text_features,
                    'logit_scale': self.logit_scale.exp()}

        ret = self.cl_head(features)

        return ret
    


class MergeLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, visual_feats, text_feats, key_padding_mask=None):
        visual_cls = visual_feats[:, 0]
        text_cls = text_feats[:, 0]
        query = query + visual_cls.unsqueeze(1) + text_cls.unsqueeze(1)
        kv = torch.cat([visual_feats, text_feats], dim=1)

        # 交叉注意力
        attn_output, _ = self.cross_attn(query=query, key=kv, value=kv, key_padding_mask=key_padding_mask)
        # 残差连接+层归一化
        query = self.norm1(query + attn_output)
        # 前馈网络
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)
        return query


class Modality_Mergerv3(nn.Module):
    def __init__(self, d_model: int = 512, d_output: int = 256, nhead: int = 8, layer_num: int = 3):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model))
        self.layers = nn.ModuleList([
            MergeLayer(d_model, nhead) for _ in range(layer_num)
        ])
        self.proj_d512 = nn.Linear(d_model, 512)
        self.proj_d256 = nn.Linear(d_model, 256)
        self.proj_d128 = nn.Linear(d_model, 128)
        self.proj_d64 = nn.Linear(d_model, 64)
        self.proj_d32 = nn.Linear(d_model, 32)

    def forward(self, visual_feats,
                text_feats, text_mask,
                cate_feats, cate_mask,
                c2c_feats,
                ):
        bs = visual_feats.size(0)

        # 创建视觉特征的全1 mask（假设视觉特征无pad）
        visual_mask = torch.ones(visual_feats.size()[:2], dtype=text_mask.dtype, device=text_mask.device)
        c2c_mask = torch.ones(c2c_feats.size()[:2], dtype=text_mask.dtype, device=text_mask.device)
        key_padding_mask = torch.cat([visual_mask, text_mask, cate_mask, c2c_mask], dim=1)
        key_padding_mask = key_padding_mask == 0

        text_feats = torch.cat([text_feats, cate_feats, c2c_feats], dim=1)

        query = self.q.expand(bs, -1, -1)
        for layer in self.layers:
            query = layer(query, visual_feats, text_feats, key_padding_mask)

        query = query.squeeze(1)

        mm_features_d512 = self.proj_d512(query)
        mm_features_d256 = self.proj_d256(query)
        mm_features_d128 = self.proj_d128(query)
        mm_features_d64 = self.proj_d64(query)
        mm_features_d32 = self.proj_d32(query)
        return mm_features_d512, mm_features_d256, mm_features_d128, mm_features_d64, mm_features_d32
