#!/bin/bash

# Variant of runs/speedrun.sh that pretrains on FineWeb's October/November/December
# 2024 CommonCrawl snapshots (CC-MAIN-2024-42, CC-MAIN-2024-46, CC-MAIN-2024-51)
# instead of the default ClimbMix-400B dataset.
# It is designed to run on a blank 8XH100 GPU node.

# 1) Example launch (simplest):
# bash runs/fineweb.sh
# 2) Example launch in a screen session:
# screen -L -Logfile runs/fineweb.log -S fineweb bash runs/fineweb.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=fineweb screen -L -Logfile runs/fineweb.log -S fineweb bash runs/fineweb.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NUM_GPU=2
export NANOCHAT_BASE_DIR="$HOME/scratch/safeLLM/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies, including dev deps (needed for the `datasets` library below)
uv sync --extra gpu --group dev
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Dataset: build our own shards from FineWeb Oct/Nov/Dec 2024, into
# $NANOCHAT_BASE_DIR/base_data_fineweb_2024q4 (this is what nanochat/dataset.py's
# DATA_DIR now points at). See dev/repackage_fineweb_2024q4.py for details.
#
# Unlike speedrun.sh's shard downloader, this streams+shuffles straight from
# HuggingFace and writes shards directly (no atomic temp-file rename), so we run
# it to completion up front rather than racing it against tokenizer training.

#python dev/repackage_fineweb_2024q4.py

# -----------------------------------------------------------------------------
# Tokenizer

# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
torchrun --standalone --nproc_per_node=$NUM_GPU -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=$NUM_GPU -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT and eval the model
torchrun --standalone --nproc_per_node=$NUM_GPU -m scripts.chat_sft -- --device-batch-size=16 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NUM_GPU -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web
