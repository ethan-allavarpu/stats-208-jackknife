#! /bin/bash
        
# Clean and process data files
python src/process-data.py \
        --crime_data_path data/raw/communities/communities.data \
        --crime_docs_path data/raw/communities/communities.names \
        --crime_out_path data/processed/crime.csv  \
        --blog_data_path data/raw/BlogFeedback/blogData_train.csv \
        --blog_out_path data/processed/blog.csv \
        --meps_data_path data/raw/h192.ssp \
        --meps_out_path data/processed/meps.csv \
        --cofi_bottle_path data/raw/CalCOFI_Database_194903-202001_csv_22Sep2021/194903-202001_Bottle.csv \
        --cofi_cast_path data/raw/CalCOFI_Database_194903-202001_csv_22Sep2021/194903-202001_Cast.csv \
        --cofi_out_path data/processed/cofi.csv
