#!/usr/bin/env python3
"""
PEFT-Bench – multi-adapter evaluation (lm-eval-harness ≥ 0.4)

• Reads every YAML manifest in ./manifests  *or* adapters.yaml
• For each manifest:
    – loads base model + PEFT adapter
    – merges LoRA/QLoRA weights
    – runs lm-eval tasks through HFLM wrapper
• Appends all results to data/peft_bench.parquet on the Hub
"""

import datetime, os, subprocess, tempfile
from pathlib import Path

import pandas as pd, yaml
from huggingface_hub import HfApi, login, hf_hub_download
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ───────────────────────────────
# 0. Load every manifest
# ───────────────────────────────
CONFIGS: list[dict] = []

if Path("adapters.yaml").exists():
    CONFIGS.extend(yaml.safe_load(open("adapters.yaml"))["adapters"])

for yml in Path("manifests").glob("*.yaml"):
    CONFIGS.append(yaml.safe_load(open(yml)))

if not CONFIGS:
    raise RuntimeError("No adapter configs found in adapters.yaml or manifests/")

# ───────────────────────────────
# 1. Authenticate to Hub
# ───────────────────────────────
token = os.getenv("HF_TOKEN")
if not token or token == "***":
    raise RuntimeError("HF_TOKEN secret is missing in this workflow run.")
login(token)

DATASET_REPO = os.environ["HF_DATASET_REPO"]          # e.g. Mdrnfox/peft-bench-metrics
api = HfApi()

# ───────────────────────────────
# 2. Evaluate every adapter
# ───────────────────────────────
all_rows = []
METRICS_TO_KEEP = {"acc", "accuracy", "acc_stderr", "f1", "exact_match"}

for cfg in CONFIGS:
    base_model_id   = cfg["base_model"]
    adapter_repo    = cfg["adapter_repo"]
    adapter_type    = cfg.get("adapter_type", "LoRA")
    tasks           = cfg["tasks"]

    tokenizer   = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    base_model  = AutoModelForCausalLM.from_pretrained(base_model_id)
    peft_model  = PeftModel.from_pretrained(base_model, adapter_repo)
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    with tempfile.TemporaryDirectory() as td:
        merged_model.save_pretrained(td)
        tokenizer.save_pretrained(td)

        hf_lm = HFLM(pretrained=td, batch_size=1, device="cpu")
        res   = evaluator.simple_evaluate(model=hf_lm, tasks=tasks)

    meta = {
        "model_id": adapter_repo,
        "adapter_type": adapter_type,
        "trainable_params": cfg.get("trainable_params"),
        "peak_gpu_mem_mb": None,
        "run_date": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode(),
    }

    for task, scores in res["results"].items():
        for metric, value in scores.items():
            if metric not in METRICS_TO_KEEP:
                continue
            all_rows.append({**meta, "task": task, "metric": metric, "value": value})

# ───────────────────────────────
# 3. Append to Parquet on Hub
# ───────────────────────────────
df_new = pd.DataFrame(all_rows)

with tempfile.TemporaryDirectory() as tmp:
    current_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="data/peft_bench.parquet",
        repo_type="dataset",
        cache_dir=tmp,
        local_dir=tmp,
        local_dir_use_symlinks=False,
    )
    df_existing = pd.read_parquet(current_path)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    df_combined = (
        df_combined
        .sort_values("run_date") 
        .drop_duplicates(
            subset=["model_id", "task", "metric"], keep="last"
        )
    )

    df_combined["value"] = pd.to_numeric(df_combined["value"], errors="coerce")

    out = Path("peft_bench.parquet")
    df_combined.to_parquet(out, index=False)

    api.upload_file(
        path_or_fileobj=out,
        path_in_repo="data/peft_bench.parquet",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Add {len(CONFIGS)} new adapter run(s)",
    )
