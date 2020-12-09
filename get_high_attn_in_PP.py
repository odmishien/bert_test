import argparse
import glob
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import japanize_matplotlib
import os


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, help="Target directory.")
    parser.add_argument("--target_attn_layer", type=str,
                        help="Target Attention Layer.")
    return parser.parse_args()


def get_target_files(target_dir, target_attn_layer):
    PP_target_files = glob.glob(
        f'{target_dir}/**/PP_attn_{target_attn_layer}.tsv', recursive=True)
    NN_target_files = glob.glob(
        f'{target_dir}/**/NN_attn_{target_attn_layer}.tsv', recursive=True)
    return PP_target_files, NN_target_files


def plot(d, path):
    fpath = "./ipag.ttf"
    wordcloud = WordCloud(
        font_path=fpath, colormap='gist_ncar').generate_from_frequencies(d)
    fig = plt.figure(figsize=(15, 12))
    plt.imshow(wordcloud)
    plt.axis("off")
    fig.savefig(path)


if __name__ == "__main__":
    args = parse_arg()
    print("1. get target files")
    PP_target_files, NN_target_files = get_target_files(
        args.target_dir, args.target_attn_layer)
    print(f'PP: {len(PP_target_files)}, NN: {len(NN_target_files)} files found.')

    PP_file_contents = []
    for target in PP_target_files:
        with open(target, 'r') as f:
            lines = pd.read_table(target, names=('attention', 'word'))
            PP_file_contents.append(lines)
    PP_concat_df = pd.concat(PP_file_contents)

    NN_file_contents = []
    for target in NN_target_files:
        with open(target, 'r') as f:
            lines = pd.read_table(target, names=('attention', 'word'))
            NN_file_contents.append(lines)
    NN_concat_df = pd.concat(NN_file_contents)

    PP_attn_mean_by_words_df = pd.DataFrame(
        index=[], columns=['word', 'attn_mean', 'counts'])
    uniq_words = PP_concat_df.word.unique()
    value_counts = PP_concat_df.word.value_counts()
    for w in uniq_words:
        df = PP_concat_df.query(f'word == "{w}"')
        record = pd.Series([w, df['attention'].mean(), value_counts.get(w)],
                           index=PP_attn_mean_by_words_df.columns)
        PP_attn_mean_by_words_df = PP_attn_mean_by_words_df.append(
            record, ignore_index=True)
    PP_attn_mean_by_words_df = PP_attn_mean_by_words_df.query("counts > 20")

    NN_attn_mean_by_words_df = pd.DataFrame(
        index=[], columns=['word', 'attn_mean', 'counts'])
    uniq_words = NN_concat_df.word.unique()
    value_counts = NN_concat_df.word.value_counts()
    for w in uniq_words:
        df = NN_concat_df.query(f'word == "{w}"')
        record = pd.Series([w, df['attention'].mean(), value_counts.get(w)],
                           index=NN_attn_mean_by_words_df.columns)
        NN_attn_mean_by_words_df = NN_attn_mean_by_words_df.append(
            record, ignore_index=True)
    NN_attn_mean_by_words_df = NN_attn_mean_by_words_df.query("counts > 20")

    PP_high_words = []
    # NNにある場合は 差が 0.2 以上あるかどうか
    # NNにない場合は 0.2　以上あるか
    for index, row in PP_attn_mean_by_words_df.iterrows():
        if not NN_attn_mean_by_words_df[NN_attn_mean_by_words_df["word"] == row["word"]].empty:
            if NN_attn_mean_by_words_df[NN_attn_mean_by_words_df["word"] == row["word"]]["attn_mean"].iloc[-1] - row["attn_mean"] < -0.2:
                PP_high_words.append(row)
            else:
                if row["attn_mean"] > 0.2:
                    PP_high_words.append(row)

    wordcloud_dict = {}
    for w in PP_high_words:
        wordcloud_dict[w['word']] = w['attn_mean']
    print('2.plotting...')
    plot(wordcloud_dict,
         f'./results/wordcloud/high_attn_in_PP_{args.target_attn_layer}.png')
