import argparse
import glob
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import japanize_matplotlib


def get_unique_value(pos_words, nega_words):
    intersection = set(pos_words) & set(nega_words)
    return list(intersection ^ set(pos_words)), list(intersection ^ set(nega_words))


def plot(df, title):
    seaborn.barplot(data=df, x='word', y='attn_mean',
                    order=df.sort_values('attn_mean', ascending=False)['word'][:45]).set_title(title)
    plt.show()


if __name__ == "__main__":
    pos_df = pd.read_csv(f'attn_mean_20201128_by_words_PP_all.csv')
    nega_df = pd.read_csv(f'attn_mean_20201128_by_words_NN_all.csv')
    pos_words = pos_df['word'].to_list()
    nega_words = nega_df['word'].to_list()
    pu_words, nu_words = [], []

    # どちらかに3倍以上出てたらuniqueとする
    for pw in pos_words:
        try:
            if pos_df[pos_df['word'] == pw]['counts'].values[0] > nega_df[nega_df['word'] == pw]['counts'].values[0] * 3:
                pu_words.append(pw)
        except:
            pass

    for nw in nega_words:
        try:
            if nega_df[nega_df['word'] == nw]['counts'].values[0] > pos_df[pos_df['word'] == nw]['counts'].values[0] * 3:
                nu_words.append(nw)
        except:
            pass

    pos_uniq_df = pd.DataFrame(
        index=[], columns=['word', 'attn_mean', 'counts'])
    for pu_word in pu_words:
        pos_uniq_se = pos_df[pos_df['word'] == pu_word]
        pos_uniq_df = pos_uniq_df.append(
            pos_uniq_se, ignore_index=True)

    nega_uniq_df = pd.DataFrame(
        index=[], columns=['word', 'attn_mean', 'counts'])
    for nu_word in nu_words:
        nega_uniq_se = nega_df[nega_df['word'] == nu_word]
        nega_uniq_df = nega_uniq_df.append(
            nega_uniq_se, ignore_index=True)
    pos_uniq_df = pos_uniq_df.query('counts > 10')
    nega_uniq_df = nega_uniq_df.query('counts > 10')
    plot(pos_uniq_df, 'pos_unique')
    plot(nega_uniq_df, 'nega_unique')
