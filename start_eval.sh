#!/bin/bash

CONFIG_PATH="conf/config.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found"
    exit 1
fi

echo ""
echo "============================================================"
echo "                    Start to evaluate                       "
echo "============================================================"
echo ""

if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found"
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ""

python eval.py
