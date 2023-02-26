#! /bin/bash

python src/replicate-trials.py \
        --input_path data/processed/crime.csv \
        --model_type nn \
        --interval_type naive

python src/replicate-trials.py \
        --input_path data/processed/crime.csv \
        --model_type nn \
        --interval_type jackknife

python src/replicate-trials.py \
        --input_path data/processed/crime.csv \
        --model_type nn \
        --interval_type jackknife_plus
