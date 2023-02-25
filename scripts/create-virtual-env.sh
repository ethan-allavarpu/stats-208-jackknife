#! /bin/bash

# Set up environment
conda update -n base -c defaults conda
conda env create -−file=environment.yml
conda activate jackknife