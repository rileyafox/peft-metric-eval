#!/usr/bin/env python3
"""
Minimal PEFT-Bench evaluation script.
Loads a base model + adapter, runs lm-eval-harness tasks, and appends results
to the peft-bench-metrics Parquet on Hugging Face.
"""
import datetime, os, subprocess, tempfile, yaml, pandas as pd
from huggingface_hub import login, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from lm_eval import evaluator

# -------- 1. load manifest --------------------------------------------------
with open("peft_bench.yaml") as f:
    cfg = yaml.safe_load(f)

BASE_MODEL   = cfg["base_model"]
ADAPTER_REPO = cfg["adapter_repo"]
TASKS        = cfg["tasks"]
ADAPTER_TYPE = cfg.get("adapter_type", "LoRA")

# -------- 2. load model & tokenizer ----------------------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER_REPO)
model.eval()

# -------- 3. run evaluation -------------------------------------------------
results = evaluator.simple_evaluate(
    model=model,
    tokenizer=tokenizer,
    tasks=TASKS,
    batch_size=4,
    device="cuda"
)

# -------- 4. flatten to DataFrame ------------------------------------------
rows = []
meta = {
    "model_id":      ADAPTER_REPO,
    "adapter_type":  ADAPTER_TYPE,
    "trainable_params": cfg.get("trainable_params"),
    "peak_gpu_mem_mb": None,
    "run_date":      datetime.datetime.utcnow().isoformat(timespec="seconds"),
    "commit_sha":    subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
}
for task, task_dict in results["results"].items():
    for metric, value in task_dict.items():
        rows.append({**meta, "task": task, "metric": metric, "value": value})
df = pd.DataFrame(rows)

# -------- 5. append to parquet on HF Hub -----------------------------------
login()                        # uses HF_TOKEN env var inside CI
HF_DATASET_REPO = os.environ["HF_DATASET_REPO"]
api = HfApi()

with tempfile.TemporaryDirectory() as tmp:
    local = api.hf_hub_download(repo_id=HF_DATASET_REPO,
                                filename="data/peft_bench.parquet",
                                repo_type="dataset",
                                cache_dir=tmp, local_dir=tmp,
                                local_dir_use_symlinks=False)
    current = pd.read_parquet(local)
    combined = pd.concat([current, df], ignore_index=True)
    combined.to_parquet("peft_bench.parquet", index=False)

    api.upload_file(
        path_or_fileobj="peft_bench.parquet",
        path_in_repo="data/peft_bench.parquet",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Add results for {ADAPTER_REPO}"
    )
