#! /bin/bash
        
# Clean and process data files
python src/process-data.py \
        --crime_data_path data/raw/communities/communities.data \
        --crime_docs_path data/raw/communities/communities.names \
        --crime_out_path data/processed/crime.csv  \
        --blog_data_path data/raw/BlogFeedback/blogData_train.csv \
        --blog_out_path data/processed/blog.csv \
        --meps_data_path data/raw/h192.ssp \
        --meps_out_path data/processed/meps.csv
