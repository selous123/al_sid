import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # Optional: use mirror for faster download in some regions
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AL-GR/AL-GR-Tiny",
    repo_type="dataset",  
    local_dir="data/AL-GR-Tiny",
    local_dir_use_symlinks=False
)