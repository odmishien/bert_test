import argparse
import glob
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import japanize_matplotlib


def get_common_value(pos_words, nega_words):
    intersection = set(pos_words) & set(nega_words)
    return list(intersection)


def plot(df):
    seaborn.barplot(data=df, x='word', y='attn_mean',
                    order=df.sort_values('attn_mean', ascending=False)['word'][:45])
    plt.show()


if __name__ == "__main__":
    pos_df = pd.read_csv(f'./results/attn_mean_20201111_by_words_PP.csv')
    nega_df = pd.read_csv(f'./results/attn_mean_20201111_by_words_NN.csv')
    pos_words = pos_df['word'].to_list()
    nega_words = nega_df['word'].to_list()
    common_words = get_common_value(pos_words, nega_words)

    pos_common_df = pd.DataFrame(
        index=[], columns=['word', 'attn_mean', 'counts'])

    nega_common_df = pd.DataFrame(
        index=[], columns=['word', 'attn_mean', 'counts'])
    for word in common_words:
        pos_se = pos_df[pos_df['word'] == word]
        pos_common_df = pos_common_df.append(pos_se, ignore_index=True)
        nega_se = nega_df[nega_df['word'] == word]
        nega_common_df = nega_common_df.append(nega_se, ignore_index=True)
    plot(pos_common_df)
    plot(nega_common_df)
