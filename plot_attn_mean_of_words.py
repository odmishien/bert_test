import argparse

import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import japanize_matplotlib


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_freq", type=str,
                        help="min_freq")
    parser.add_argument("--target_lp", type=str,
                        help="Target label and prediction pair.")
    return parser.parse_args()


def plot(df):
    seaborn.barplot(data=df, x='word', y='attn_mean',
                    order=df.sort_values('attn_mean', ascending=False)['word']).set_xticklabels(rotation=90)
    plt.show()


if __name__ == "__main__":
    args = parse_arg()
    df = pd.read_csv(f'./results/attn_mean_by_words_{args.target_lp}.csv')
    df = df.query(f'counts > {args.min_freq}')
    plot(df)
