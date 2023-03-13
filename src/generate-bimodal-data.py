import argparse
import numpy as np
import pandas as pd
import scipy.stats

np.random.seed(208)

argp = argparse.ArgumentParser()
argp.add_argument("--N", type=int, required=True)
argp.add_argument("--p", type=int, required=True)
argp.add_argument("--out_path", type=str, required=True)
args = argp.parse_args()

if __name__ == "__main__":
    X = np.random.exponential(scale=0.1, size=(args.N, args.p))
    beta = np.random.randint(0, 10, size=args.p)
    noise = np.random.normal(loc=10, size=args.N) * np.random.choice(
        [-1, 1], size=args.N
    )
    print(f"Beta: {beta}")
    Y = X @ beta + noise

    pd.concat([pd.DataFrame(X), pd.Series(Y)], axis=1).to_csv(
        args.out_path, index=False
    )

    Y_hat = X @ beta
    margin = 10 + scipy.stats.norm.ppf(1 - 0.10)
    covered = (Y_hat - margin <= Y) & (Y <= Y_hat + margin)
    print(f"Coverage Rate: {covered.mean()}")
    print(f"Coverage Width: {np.round(2 * margin, 2)}")
