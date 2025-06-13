#!/usr/bin/env python3
"""
PEFT-Bench evaluation script 
--------------------------------------------------------------

• Loads a base model + LoRA / QLoRA adapter (via PEFT)
• Wraps them in lm-eval's HFLM wrapper (new API)
• Runs lm-eval tasks and appends the results to the Parquet leaderboard
"""

import datetime
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import HfApi, login
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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
# 2. Load tokenizer & base model, then attach adapter
# ─────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
model.eval()

# ─────────────────────────────────────────────────────────────
# 3. Wrap in the *new* lm-eval Hugging Face wrapper
#    - `pretrained` is REQUIRED even if you give model+tokenizer objects
# ─────────────────────────────────────────────────────────────
hf_lm = HFLM(
    pretrained="unused",        # any non-null string satisfies the arg
    model=BASE_MODEL,
    tokenizer=tokenizer,
    batch_size=1,               # keep tiny for CPU runner
    device="cpu",
)

# ─────────────────────────────────────────────────────────────
# 4. Run evaluation
# ─────────────────────────────────────────────────────────────
results = evaluator.simple_evaluate(
    model=hf_lm,
    tasks=TASKS,
)

# ─────────────────────────────────────────────────────────────
# 5. Flatten results → DataFrame
# ─────────────────────────────────────────────────────────────
rows = []
meta = {
    "model_id": ADAPTER_REPO,
    "adapter_type": ADAPTER_TYPE,
    "trainable_params": cfg.get("trainable_params"),
    "peak_gpu_mem_mb": None,
    "run_date": datetime.datetime.utcnow().isoformat(timespec="seconds"),
    "commit_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).strip().decode(),
}
for task, scores in results["results"].items():
    for metric, value in scores.items():
        rows.append({**meta, "task": task, "metric": metric, "value": value})

df = pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# 6. Append to Parquet on the Hub
# ─────────────────────────────────────────────────────────────
login()                                    # reads HF_TOKEN env var
DATASET_REPO = os.environ["HF_DATASET_REPO"] 
api = HfApi()

with tempfile.TemporaryDirectory() as tmpdir:
    current_path = api.hf_hub_download(
        repo_id=DATASET_REPO,
        filename="data/peft_bench.parquet",
        repo_type="dataset",
        cache_dir=tmpdir,
        local_dir=tmpdir,
        local_dir_use_symlinks=False,
    )
    existing = pd.read_parquet(current_path)
    combined = pd.concat([existing, df], ignore_index=True)
    combined.to_parquet("peft_bench.parquet", index=False)

    api.upload_file(
        path_or_fileobj="peft_bench.parquet",
        path_in_repo="data/peft_bench.parquet",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Add results for {ADAPTER_REPO}",
    )
