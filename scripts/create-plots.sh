#! /bin/bash
        
# Replicate plots from paper
python src/generate-trial-plots.py \
    --input_dir data/processed/ \
    --output_dir visuals/ \
    --n_trials 20
