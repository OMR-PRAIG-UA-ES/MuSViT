#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Batch of training runs reproducing the paper-style experiments.
#
# Two blocks of five runs each cover all datasets:
#   * Block 1: linear probing (frozen backbone). Larger batch (16), an 8x64
#     patch grid (cols=64 -> shorter CTC time axis).
#   * Block 2: LoRA fine-tuning. Smaller batch (8), a wider 8x128 patch grid
#     (cols=128 -> longer time axis for denser scores).
#
# Common flags:
#   --model_name    backbone key from config.data_models (here: musvit)
#   --ds_name       dataset key from config.data_paths
#   --method        linear_prob | lora
#   --batch_size    mini-batch size
#   --start_eval    first epoch to run validation / checkpointing
#   --shape_patches "<rows> <cols>" patch grid
#   --lr            Adam learning rate
#
# Run all sequentially with:  bash executions.sh
# (or copy/paste individual lines to run a single experiment).
# ---------------------------------------------------------------------------

# --- Linear probing (backbone frozen) -------------------------------------
python train.py --model_name=musvit --ds_name=catedrales --method=linear_prob --batch_size=16 --start_eval 20 --shape_patches 8 64 --lr=0.0003
python train.py --model_name=musvit --ds_name=capitan --method=linear_prob --batch_size=16 --start_eval 20 --shape_patches 8 64 --lr=0.0003
python train.py --model_name=musvit --ds_name=fmt --method=linear_prob --batch_size=16 --start_eval 20 --shape_patches 8 64 --lr=0.0003
python train.py --model_name=musvit --ds_name=seils --method=linear_prob --batch_size=16 --start_eval 20 --shape_patches 8 64 --lr=0.0003
python train.py --model_name=musvit --ds_name=guatemala --method=linear_prob --batch_size=16 --start_eval 20 --shape_patches 8 64 --lr=0.0003

# --- LoRA fine-tuning (adapters on q/k/v) ---------------------------------
python train.py --model_name=musvit --ds_name=catedrales --method=lora --batch_size=8 --start_eval 20 --shape_patches 8 128 --lr=0.0003
python train.py --model_name=musvit --ds_name=capitan --method=lora --batch_size=8 --start_eval 20 --shape_patches 8 128 --lr=0.0003
python train.py --model_name=musvit --ds_name=fmt --method=lora --batch_size=8 --start_eval 20 --shape_patches 8 128 --lr=0.0003
python train.py --model_name=musvit --ds_name=seils --method=lora --batch_size=8 --start_eval 20 --shape_patches 8 128 --lr=0.0003
python train.py --model_name=musvit --ds_name=guatemala --method=lora --batch_size=8 --start_eval 20 --shape_patches 8 128 --lr=0.0003
