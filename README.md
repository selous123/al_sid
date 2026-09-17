<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/selous123/al_sid">
    <img src="asset/Title.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">FORGE</h3>

  <p align="center">
     <b>FO</b>rming semantic identifie<b>R</b>s for <b>G</b>enerative retri<b>E</b>val in Industrial Datasets
    <br />
    <a href="https://huggingface.co/datasets/AL-GR"><strong>Explore full dataset in Huggingface »</strong></a>
    <br />
    <br />
    <a href="https://github.com/selous123/al_sid">View Demo</a>
    &middot;
    <a href="https://github.com/selous123/al_sid/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/selous123/al_sid/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>




<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://github.com/selous123/al_sid)

Semantic identifiers (SIDs) have gained increasing interest in generative retrieval (GR) due to their meaningful semantic discriminability. Existing studies typically rely on arbitrarily defined SIDs while neglecting the influence of SID configurations on GR. Besides, evaluations conducted on datasets with limited multimodal features and behaviors also hinder their reliability in industrial traffic. To address these limitations, we propose **FORGE**, a comprehensive benchmark for <u>**FO**</u>rming semantic identifie<u>**R**</u> in <u>**G**</u>enerative <u>**E**</u>trieval with industrial datasets. Specifically, FORGE is equipped with a dataset comprising **14 billion** user interactions and multi-modal features of **250 million** items collected from an e-commerce platform, enabling researchers to construct and evaluate their own SIDs.
Leveraging this dataset, we systematically explore various strategies for SID generation and validate their effectiveness across different settings and tasks. Extensive online experiments show **8.93\%** and **0.35\%** improvements in PVR and transaction count, highlighting the practical value of our approach. 
Notably, we propose two novel metrics of SID that correlate well with GR performance, providing insights into a convenient measurement of SID quality without training GR. Subsequent offline pretraining also offers support for online convergence in industrial applications. 
The code and data are available at [code repo](https://github.com/selous123/al_sid).


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
1. Clone the repository:
   ```bash
   git clone repo_name
   cd repo
   ```
2. Install dependencies:
  * Python 3.8+
  * PyTorch 1.10+
  * requirements.txt
    ```sh
    pip install -r requirements.txt
    ```

<!-- ABOUT THE Dataset-->
### Dataset Decription(Demo)
1. For SID Generation Task:
```bash
wget -P datas/ https://mvap-public-data.oss-cn-zhangjiakou.aliyuncs.com/ICLR_2026_data/reconstruct_data_mask.npz
wget -P datas/ https://mvap-public-data.oss-cn-zhangjiakou.aliyuncs.com/ICLR_2026_data/contrastive_data_mask.npz
```

Dataset Preview:
* contrastive_data_mask.npz for contrastive task in Eq.2

The file contains three components: 

(1) a deduplicated mapping table between item IDs and their corresponding indices ("itemEncId") with shape (6,844,930, 2), where each row is an [item_id, index] pair (e.g., [855036080309, 0]); 

(2) a list of item pairs ("pairs") with shape (9,509,084, 2), representing co-occurrence or association relationships between items (e.g., [855036080309, 545092516562]); and 

(3) a deduplicated embedding matrix with shape (6,844,930, 512), where each row is a 512-dimensional vector representation of an item (e.g., [xx, xx,..., xx]). 
```python
import numpy as np
import os

# 1. load .npz file
file_path = os.path.join(dirpath, filename2)  # replace your filename
data = np.load(file_path, allow_pickle=True)
itemEncID, pairs, embeds = data['itemEncID'].item(), data['pairs'], data['embeds'].astype(np.float32)
for key, item in itemEncID.items():
    print(key, item) #[855036080309, 0], [545092516562, 1]
    if item > 100:
        break
print("pairs:", pairs.shape, pairs[:1]) #(9509084, 2) [[855036080309 545092516562]]
print("embeds:", embeds.shape, embeds[:1]) #(6844930, 512) [[xx,xx,...,xx]]
```


* reconstruct_data_mask.npz for reconstruction task in Eq.3
``` python
import numpy as np
import os
# 1. load .npz file
dirpath = '~/git/al_sid/SID_generation/datas'
filename1 = 'reconstruct_data_mask.npz'
file_path = os.path.join(dirpath, filename1)  # replace your filename1
data = np.load(file_path)
print("Available arrays:", data.files) ##Available arrays: ['ids', 'embeds']
for key in data:
    print(f"{key}: {data[key].shape}")
    print(data[key][:1])
data.close()
#ids: (4148316,) [813799260043]
#embeds: (4148316, 512) [xx,xx,...,xx]
```


2. Seq Data for Generative Task:
```python
from datasets import load_dataset
# Login using e.g. `huggingface-cli login` to access this dataset
dataset = load_dataset("AL-GR/AL-GR-Tiny", data_files="train_data/s1_tiny.csv", split="train")
```

Data Availability: All training datasets have been released on Hugging Face.

---
| **Codebook** | **final (Ours)** | **base (Base)** |
| :--- | :--- | :--- |
| **Training/Test Dataset** | [AL-GR / AL-GR-v1](https://huggingface.co/datasets/AL-GR/AL-GR-v1/) | [AL-GR / AL-GR](https://huggingface.co/datasets/AL-GR/AL-GR/) |
| **Item-SID Mapping** | [AL-GR/Item-SID/sid_final.csv](https://huggingface.co/datasets/AL-GR/Item-SID/blob/main/sid_final.csv) | [AL-GR/Item-SID/sid_base.csv](https://huggingface.co/datasets/AL-GR/Item-SID/blob/main/sid_base.csv) |
| **Tiny Dataset** | Both training and testing use the final version: [AL-GR / AL-GR-Tiny](https://huggingface.co/datasets/AL-GR/AL-GR-Tiny) | Base version without preprocessing; if needed, can be obtained by joining [AL-GR/AL-GR-Tiny/origin_behavior](https://huggingface.co/datasets/AL-GR/AL-GR-Tiny/tree/main/origin_behavior) from the Tiny data with [the base version Item-SID](https://huggingface.co/datasets/AL-GR/AL-GR-Tiny/blob/main/item_info/tiny_item_sid_base.csv) mapping. |
| **ITEM-EMB Multimodal Data** | final version: [AL-GR / Item-EMB / final_feature](https://huggingface.co/datasets/AL-GR/Item-EMB/tree/main/final_feature) | base version: [AL-GR / Item-EMB / base_feature](https://huggingface.co/datasets/AL-GR/Item-EMB/tree/main/base_feature) |
---

Use this code for preview the ITEM-EMB Multimodal Data:
```python
import base64
import numpy as np
from datasets import load_dataset
import pandas as pd
import logging

data_file = "./data/part_0.csv"
chunk_size = 50  #
INPUT_EMBEDDING_COL = 'feature'  # the name of embedding col: "feature"
INPUT_ID_COL = 'base62_string'  # the name of item_id after hash: "base62_string"
item_ids_buffer = []
embeddings_buffer = []
EXPECTED_EMBEDDING_DIM = 512
for chunk in pd.read_csv(data_file, chunksize=chunk_size, encoding='utf-8'):
    chunk = chunk[chunk[INPUT_EMBEDDING_COL] != INPUT_EMBEDDING_COL]
    if chunk.empty:
        continue
    item_ids_buffer.extend(chunk[INPUT_ID_COL].tolist())
    embeddings_buffer.extend(chunk[INPUT_EMBEDDING_COL].tolist())
    break

# print(item_ids_buffer, embeddings_buffer[:5])
embedding_list = []
valid_item_ids = []
for idx, (item_id, embedding_str) in enumerate(zip(item_ids_buffer, embeddings_buffer)):
    try:
        embedding_np = np.frombuffer(base64.b64decode(embedding_str), dtype=np.float32)
        if embedding_np.shape[0] != EXPECTED_EMBEDDING_DIM:
            logging.warning(
                f"The dim of item ID '{item_id}' is not correct"
                f"Expected dim: {EXPECTED_EMBEDDING_DIM}, Actual dim: {embedding_np.shape[0]}。"
            )
            continue
        embedding_list.append(embedding_np)
        valid_item_ids.append(item_id)
        print(f"idx:{idx}, item_id:{item_id}, embedding:{embedding_np[:5]}")
    except Exception as e:
        logging.warning(f"Error occurs when decoding item id '{item_id}'")
        continue
```

### SID Generation
1. Training the Model

To start distributed training, use the following command:
```bash
python -m torch.distributed.launch --nnodes=2 --nproc_per_node=1 --master_port=27646 train.py --output_dir=/path/to/output --save_prefix=MODEL_NAME --cfg=configs/rqvae_i2v.yml
```

2. Parameters
- `--cfg`: Path to the configuration file.
- `--output_dir`: Directory for model outputs.
- `--save_prefix`: Prefix for saving the model.

3. Testing the Model

Use the following command to start testing:

```bash
python infer_SID.py
```

### Generative Retrival

1. Clone the repo
   ```sh
   git clone repo_name
   cd repo_name/algr
   ```
2. training scripts:
    ```
    python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 runner.py --config=config/qwen2.5_05b_3layer_s1.json
    ```
3. predict scripts:
    ```
    python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 runner.py --config=config/generate_qwen2.5_05b_3layer_tiny.json
    ```
4. calculate Hitrate:
    ```
    # nebula test: python calc_hr.py --dataset_name=/home/admin/.cache/huggingface/modules/datasets_modules/datasets/AL-GR--AL-GR-Tiny/25dea07242891a2d --nebula
    python calc_hr.py --item_sid_file=item_info/tiny_item_sid_final.csv --generate_file=logs/generate_qwen2.5_05b_3layer_tiny/output.jsonl --decoder_only
    python calc_hr.py --item_sid_file=item_info/tiny_item_sid_final.csv --generate_file=logs/generate_t5base_3layer_tiny/output.jsonl
    ```
5. Running in the background
    ```
    nohup python -m torch.distributed.launch --nnodes=1 --nproc_per_node=16 runner.py --config=config/t5base_3layer_s1.json  >logs/t5base_3layer_s1/log.txt 2>&1 &
    nohup python -m torch.distributed.launch --nnodes=1 --nproc_per_node=16 runner.py --config=config/qwen2.5_05b_3layer_s1.json  >logs/qwen2.5_05b_3layer_s1/log.txt 2>&1 &
    ```



<!-- ROADMAP -->
## Roadmap

- [x] Generative Retrieval Training
- [x] SID Generation
- [x] Data Processing

See the [open issues](https://github.com/selous123/al_sid/issues) for a full list of proposed features (and known issues).

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Top contributors:

<a href="https://github.com/selous123/al_sid/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=selous123/al_sid" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.



<!-- CONTACT -->
## Contact
If you have any questions or encounter difficulties, we welcome you to contact us via [GitHub Issues](https://github.com/selous123/al_sid/issues). We aim to respond promptly and support you in quickly getting up and running with generative recommendation.

## Citing this work
Please cite the following paper if you find our code helpful.

```
@inproceedings{fu2026forge,
  title={Forge: Forming semantic identifiers for generative retrieval in industrial datasets},
  author={Fu, Kairui and Zhang, Tao and Xiao, Shuwen and Wang, Ziyang and Zhang, Xinming and Zhang, Chenchi and Yan, Yuliang and Zheng, Junjun and Kong, Xiangheng and Zhang, Shengyu and others},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={8858--8869},
  year={2026}
}
```



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/selous123/al_sid.svg?style=for-the-badge
[contributors-url]: https://github.com/selous123/al_sid/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/selous123/al_sid.svg?style=for-the-badge
[forks-url]: https://github.com/selous123/al_sid/network/members
[stars-shield]: https://img.shields.io/github/stars/selous123/al_sid.svg?style=for-the-badge
[stars-url]: https://github.com/selous123/al_sid/stargazers
[issues-shield]: https://img.shields.io/github/issues/selous123/al_sid.svg?style=for-the-badge
[issues-url]: https://github.com/selous123/al_sid/issues
[license-shield]: https://img.shields.io/github/license/selous123/al_sid.svg?style=for-the-badge
[license-url]: https://github.com/selous123/al_sid/blob/main/LICENSE
[product-screenshot]: asset/FORGE.png
