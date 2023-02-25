import argparse
import pandas as pd
import re

argp = argparse.ArgumentParser()
argp.add_argument("--crime_data_path", type=str, required=True)
argp.add_argument("--crime_docs_path", type=str, required=True)
argp.add_argument("--crime_out_path", type=str, required=True)
args = argp.parse_args()


def clean_crime_data(in_data: str, in_names: str, out_path: str) -> tuple:
    """
    Clean and write community crime data to include only numeric predictors.
    Also drop features with any missing observations

    Parameters
    ----------
    in_data: str
        File path for input crime data
    in_names: str
        File path for input data document with column names
    out_path: str
        File path to which we write the processed CSV file

    Returns
    -------
    Tuple with the dimensions of the trimmed dataset
    """
    crime = pd.read_csv(in_data, header=None, na_values="?")
    with open(in_names) as f:
        cols = [l.split()[1] for l in f.readlines() if re.match("@attribute", l)]
    crime.columns = cols
    # Columns specified as not predictive
    drop_cols = ["state", "county", "community", "communityname", "fold"]
    # Categorical columns
    cat_cols = [
        col
        for col, col_type in zip(crime.columns, crime.dtypes)
        if col_type not in (int, float) and col not in drop_cols
    ]
    drop_cols += cat_cols
    crime = crime.drop(columns=drop_cols).dropna(axis="columns")
    crime.to_csv(out_path, index=False)
    return crime.shape


if __name__ == "__main__":

    crime_dim = clean_crime_data(
        args.crime_data_path, args.crime_docs_path, args.crime_out_path
    )

    print(f"Crime CSV file written to {args.crime_out_path}.")
    print(f"Data has {crime_dim[0]} observations, {crime_dim[1] - 1} features")
