#! /bin/bash
        
# Generate data with bimodal noise
python src/generate-bimodal-data.py \
    --N 100000 --p 10 --out_path data/simulated/bimodal-data.csv

# Test ridge regression
python src/replicate-trials.py \
        --input_path data/simulated/bimodal-data.csv \
        --model_type ridge \
        --interval_type jackknife
