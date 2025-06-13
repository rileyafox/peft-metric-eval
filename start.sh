#!/usr/bin/env bash
set -e

echo "Starting PEFT-Bench evaluation on $(date)"
python /app/run_eval.py

echo "Completed evaluation, requesting shutdown"
curl -s -X POST "$HF_ENDPOINT_SHUTDOWN"
