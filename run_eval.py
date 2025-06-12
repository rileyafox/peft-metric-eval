#!/usr/bin/env python3
"""
Minimal PEFT-Bench evaluation script.

• Loads a base model + PEFT adapter
• Runs lm-eval-harness tasks (current API, no tokenizer kwarg)
• Appends results to the peft-bench-metrics Parquet on Hugging Face
"""

import datetime
import os
import subprocess
import tempfile

import pandas as pd
import yaml
from huggingface_hub import HfApi, login
from lm_eval import evaluator
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM


# ─────────────────────────────────────────────────────────────
# 1. Load manifest (peft_bench.yaml)
# ─────────────────────────────────────────────────────────────
with open("peft_bench.yaml") as f:
    cfg = yaml.safe_load(f)

BASE_MODEL = cfg["base_model"]        
ADAPTER_REPO = cfg["adapter_repo"]      
TASKS = cfg["tasks"]                     
ADAPTER_TYPE = cfg.get("adapter_type", "LoRA")

# ─────────────────────────────────────────────────────────────
# 2. Load tokenizer & base model
# ─────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")

# 3. Attach the adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
model.eval()

# 4. Wrap in lm-eval’s HuggingFace adapter
hf_lm = HFLM(model_name=None, model=model, tokenizer=tokenizer)

# ─────────────────────────────────────────────────────────────
# 3. Run evaluation (no tokenizer kwarg)
# ─────────────────────────────────────────────────────────────
results = evaluator.simple_evaluate(
    model=hf_lm,
    tasks=TASKS,
    batch_size=1,          
    device="cpu"
)

# ─────────────────────────────────────────────────────────────
# 4. Flatten results → DataFrame
# ─────────────────────────────────────────────────────────────
rows = []
meta = {
    "model_id": ADAPTER_REPO,
    "adapter_type": ADAPTER_TYPE,
    "trainable_params": cfg.get("trainable_params"),
    "peak_gpu_mem_mb": None,   # add later if you capture NVML
    "run_date": datetime.datetime.utcnow().isoformat(timespec="seconds"),
    "commit_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).strip().decode(),
}
for task, task_dict in results["results"].items():
    for metric, value in task_dict.items():
        rows.append({**meta, "task": task, "metric": metric, "value": value})

df = pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# 5. Append to Parquet on the Hub
# ─────────────────────────────────────────────────────────────
login()                                  
HF_DATASET_REPO = os.environ["HF_DATASET_REPO"]
api = HfApi()

with tempfile.TemporaryDirectory() as tmpdir:
    # 5a. Download current Parquet
    local_path = api.hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="data/peft_bench.parquet",
        repo_type="dataset",
        cache_dir=tmpdir,
        local_dir=tmpdir,
        local_dir_use_symlinks=False,
    )
    current = pd.read_parquet(local_path)

    # 5b. Concatenate and save
    combined = pd.concat([current, df], ignore_index=True)
    combined.to_parquet("peft_bench.parquet", index=False)

    # 5c. Push back to dataset repo
    api.upload_file(
        path_or_fileobj="peft_bench.parquet",
        path_in_repo="data/peft_bench.parquet",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Add results for {ADAPTER_REPO}",
    )
