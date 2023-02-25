#! /bin/bash

# Set up environment
conda update -n base -c defaults conda
conda env create −f environment.yml
conda activate jackknife