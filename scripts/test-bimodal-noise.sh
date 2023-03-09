#! /bin/bash
        
# Generate data with bimodal noise
python src/generate-bimodal-data.py \
    --N 100000 --p 5 --out_path data/simulated/bimodal-data.csv

# Test ridge regression
python src/replicate-trials.py \
        --input_path data/simulated/bimodal-data.csv \
        --model_type linear \
        --interval_type naive
python src/replicate-trials.py \
        --input_path data/simulated/bimodal-data.csv \
        --model_type linear \
        --interval_type jackknife
