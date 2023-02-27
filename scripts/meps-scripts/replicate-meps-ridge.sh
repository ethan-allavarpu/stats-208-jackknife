#! /bin/bash

python src/replicate-trials.py \
        --input_path data/processed/meps.csv \
        --model_type ridge \
        --interval_type naive

python src/replicate-trials.py \
        --input_path data/processed/meps.csv \
        --model_type ridge \
        --interval_type jackknife

python src/replicate-trials.py \
        --input_path data/processed/meps.csv \
        --model_type ridge \
        --interval_type cv

python src/replicate-trials.py \
        --input_path data/processed/meps.csv \
        --model_type ridge \
        --interval_type conformal
