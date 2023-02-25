#! /bin/bash
        
# Clean data files
python src/clean-data.py \
        --crime_data_path data/raw/communities/communities.data \
        --crime_docs_path data/raw/communities/communities.names \
        --crime_out_path data/processed/crime.csv