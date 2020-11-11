import argparse

import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import japanize_matplotlib


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_freq", type=str,
                        help="min_freq")
    return parser.parse_args()


def plot(df):
    seaborn.barplot(data=df, x='word', y='attn_mean',
                    order=df.sort_values('attn_mean', ascending=False)['word'])
    plt.show()


if __name__ == "__main__":
    args = parse_arg()
    df = pd.read_csv('./attn_mean_by_words_df.csv')
    df = df.query(f'counts > {args.min_freq}')
    plot(df)
