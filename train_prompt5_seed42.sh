#!/usr/bin/env bash
model_name='ProTACT'
prompt=5
seed=42
python train_ProTACT.py --test_prompt_id ${prompt} --model_name ${model_name} --seed ${seed} --num_heads 2 --features_path 'data/LDA/hand_crafted_final_'
