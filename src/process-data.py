import argparse
import numpy as np
import pandas as pd
import re

argp = argparse.ArgumentParser()
argp.add_argument("--crime_data_path", type=str, required=True)
argp.add_argument("--crime_docs_path", type=str, required=True)
argp.add_argument("--crime_out_path", type=str, required=True)
argp.add_argument("--blog_data_path", type=str, required=True)
argp.add_argument("--blog_out_path", type=str, required=True)
args = argp.parse_args()


def process_crime_data(in_data: str, in_names: str, out_path: str) -> tuple:
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
        column_names = [
            l.split()[1] for l in f.readlines() if re.match("@attribute", l)
        ]
    crime.columns = column_names
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


def process_blog_data(in_data: str, out_path: str) -> tuple:
    """
    Process BlogFeedback data and write the processed data to a CSV file.
    Only change is transforming the response: Y = log(1 + # comments)

    Parameters
    ----------
    in_data: str
        File path for input BlogFeedback data
    out_path: str
        File path to which we write the processed CSV file

    Returns
    -------
    Tuple with the dimensions of the processed dataset
    """
    blog = pd.read_csv(in_data, header=None)
    # Name columns as x or y variables
    col_names = ["x" + str(i) for i in range(blog.shape[1] - 1)] + ["y"]
    blog.columns = col_names
    # Transform y
    blog.y = blog.y.apply(lambda y: np.log(1 + y))
    blog.to_csv(out_path, index=False)
    return blog.shape


if __name__ == "__main__":
    crime_dim = process_crime_data(
        args.crime_data_path, args.crime_docs_path, args.crime_out_path
    )
    print(f"Crime CSV file written to {args.crime_out_path}.")
    print(f"Data has {crime_dim[0]} observations, {crime_dim[1] - 1} features")

    blog_dim = process_blog_data(args.blog_data_path, args.blog_out_path)
    print(f"Blog CSV file written to {args.blog_out_path}.")
    print(f"Data has {blog_dim[0]} observations, {blog_dim[1] - 1} features")