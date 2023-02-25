#! /bin/bash
        
# Replicate trials from paper
python src/replicate-trials.py \
        --crime_data data/processed/crime.csv \
        --blog_data data/processed/blog.csv \
        --out_dir data/processed/