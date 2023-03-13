#! /bin/bash

python src/replicate-trials.py \
        --input_path data/processed/cofi.csv \
        --model_type lasso \
        --interval_type naive

python src/replicate-trials.py \
        --input_path data/processed/cofi.csv \
        --model_type lasso \
        --interval_type jackknife

python src/replicate-trials.py \
        --input_path data/processed/cofi.csv \
        --model_type lasso \
        --interval_type cv

python src/replicate-trials.py \
        --input_path data/processed/cofi.csv \
        --model_type lasso \
        --interval_type conformal
