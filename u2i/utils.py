import torch
import torch.nn as nn

def count_embedding_and_dense_params(model):
    embedding_params = 0
    dense_params = 0

    for name, module in model.named_children():
        if isinstance(module, nn.Embedding):
            # Embedding 层参数量：num_embeddings * embedding_dim
            params = module.weight.data.numel()
            print(f"Embedding Layer: {name} -> {params:,} parameters")
            embedding_params += params

        elif isinstance(module, nn.Linear):
            # Dense 层参数量：weight + bias
            weight_params = module.weight.data.numel()
            bias_params = module.bias.data.numel() if module.bias is not None else 0
            total_params = weight_params + bias_params
            print(f"Dense Layer: {name} -> {total_params:,} parameters")
            dense_params += total_params

        # 如果是更复杂的嵌套结构（如 Sequential、Transformer block），递归查找
        else:
            # 递归进入子模块
            sub_embedding, sub_dense = count_embedding_and_dense_params(module)
            embedding_params += sub_embedding
            dense_params += sub_dense

    return embedding_params, dense_params


# 使用示例
if __name__ == "__main__":
    # 假设你有一个模型实例 model
    model = nn.Sequential(
        nn.Embedding(num_embeddings=10000, embedding_dim=128),
        nn.Linear(128, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )

    # 统计参数
    emb_params, dense_params = count_embedding_and_dense_params(model)

    print("\n--- Summary ---")
    print(f"Total Embedding Parameters: {emb_params:,}")
    print(f"Total Dense (Fully Connected) Parameters: {dense_params:,}")
    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
