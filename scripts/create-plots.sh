#! /bin/bash
        
# Replicate plots from paper
python src/generate_trial_plots.py \
    --input_dir data/processed/ \
    --output_dir visuals/ \
    --n_trials 20

# Create simulation plots
python src/plot-simulation.py \
    --input_dir data/simulated/ \
    --output_dir visuals/ \
    --n_trials 20