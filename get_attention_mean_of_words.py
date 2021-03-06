import argparse
import glob
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import japanize_matplotlib
import os


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_lp", type=str,
                        help="Target label and prediction pair.")
    parser.add_argument("--target_dir", type=str, help="Target directory.")
    parser.add_argument("--target_attn_layer", type=str,
                        help="Target Attention Layer.")
    return parser.parse_args()


def get_target_files(target_dir, target_attn_layer, target_lp):
    if target_lp == 'all':
        target_files = glob.glob(
            f'{target_dir}/**/**_attn_{target_attn_layer}.tsv', recursive=True)
    else:
        target_files = glob.glob(
            f'{target_dir}/**/{target_lp}_attn_{target_attn_layer}.tsv', recursive=True)
    return target_files


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
    target_files = get_target_files(
        args.target_dir, args.target_attn_layer, args.target_lp)
    print(f'{len(target_files)} files found.')
    file_contents = []
    for target in target_files:
        with open(target, 'r') as f:
            lines = pd.read_table(target, names=('attention', 'word'))
            file_contents.append(lines)
    # 読み込んだcsvを連結
    concat_df = pd.concat(file_contents)
    # csvあれば読み込み/なければ作成
    # if os.path.exists(f'./results/attn_mean_20201128_by_words_{args.target_lp}.csv'):
    #     attn_mean_by_words_df = pd.read_csv(
    #         f'./results/attn_mean_20201128_by_words_{args.target_lp}.csv')
    # else:
    attn_mean_by_words_df = pd.DataFrame(
        index=[], columns=['word', 'attn_mean', 'counts'])
    uniq_words = concat_df.word.unique()
    value_counts = concat_df.word.value_counts()
    for w in uniq_words:
        df = concat_df.query(f'word == "{w}"')
        record = pd.Series([w, df['attention'].mean(), value_counts.get(w)],
                           index=attn_mean_by_words_df.columns)
        attn_mean_by_words_df = attn_mean_by_words_df.append(
            record, ignore_index=True)
    print("2. plot attentions")
    plot_df = attn_mean_by_words_df.query('counts > 20').sort_values(
        'attn_mean', ascending=False)[:50]
    wordcloud_dict = {}
    for index, row in plot_df.iterrows():
        wordcloud_dict[row['word']] = row['attn_mean']
    plot(wordcloud_dict,
         f'results/attn_mean_png/mean_{int(args.target_attn_layer) + 1}_{args.target_lp}.png')
