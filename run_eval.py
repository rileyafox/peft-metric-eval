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
    
token = os.getenv("HF_TOKEN")
if token and token != "***":
    login(token)
else:
    raise RuntimeError("HF_TOKEN not available in this workflow run.")
DATASET_REPO = os.environ["HF_DATASET_REPO"] 
api = HfApi()


BASE_MODEL = cfg["base_model"]        
ADAPTER_REPO = cfg["adapter_repo"]
TASKS = cfg["tasks"]
ADAPTER_TYPE = cfg.get("adapter_type", "LoRA")

# ─────────────────────────────────────────────────────────────
# 2. Load tokenizer & base model, then attach adapter
# ─────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
peft_model   = PeftModel.from_pretrained(base_model, ADAPTER_REPO)

merged_model = peft_model.merge_and_unload() 
merged_model.eval()           

# ─────────────────────────────────────────────────────────────
# 3. Wrap in the *new* lm-eval Hugging Face wrapper
#    - `pretrained` is REQUIRED even if you give model+tokenizer objects
# ─────────────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmp_dir:
    merged_model.save_pretrained(tmp_dir) 
    tokenizer.save_pretrained(tmp_dir)

    hf_lm = HFLM(
        pretrained=tmp_dir,        # ← MUST be a real folder or Hub ID
        batch_size=1,              # keep tiny for CPU runner
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
METRICS_TO_KEEP = {"acc", "accuracy", "acc_stderr", "f1", "exact_match"}

meta = {
    "model_id": ADAPTER_REPO,
    "adapter_type": ADAPTER_TYPE,
    "trainable_params": cfg.get("trainable_params"),
    "peak_gpu_mem_mb": None,
    "run_date": datetime.datetime.utcnow().isoformat(timespec="seconds"),
    "commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode(),
}

rows = []
for task, score_dict in results["results"].items():
    for metric, value in score_dict.items():
        if metric not in METRICS_TO_KEEP:
            continue          # skip "alias" or other helper metrics
        rows.append({**meta, "task": task, "metric": metric, "value": value})

df_new = pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# 6. Append to Parquet on the Hub
# ─────────────────────────────────────────────────────────────
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
    combined["value"] = pd.to_numeric(combined["value"], errors="coerce")

    combined.to_parquet("peft_bench.parquet", index=False)

    api.upload_file(
        path_or_fileobj="peft_bench.parquet",
        path_in_repo="data/peft_bench.parquet",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Add results for {ADAPTER_REPO}",
    )
