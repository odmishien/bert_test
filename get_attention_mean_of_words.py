import argparse
import glob
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import japanize_matplotlib


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, help="Target directory.")
    parser.add_argument("--target_attn_layer", type=str,
                        help="Target Attention Layer.")
    return parser.parse_args()


def get_target_files(target_dir="results/masuda/attention_2000", target_attn_layer='0'):
    target_files = glob.glob(
        f'{target_dir}/**/**_attn_{target_attn_layer}.tsv', recursive=True)
    return target_files


def plot(df):
    seaborn.barplot(data=df, x='word', y='attention',
                    order=df['word'].value_counts().index[200:250])
    plt.show()


if __name__ == "__main__":
    args = parse_arg()
    print("1. get target files")
    target_files = get_target_files(
        args.target_dir, args.target_attn_layer)
    print(f'{len(target_files)} files found.')
    file_contents = []
    for target in target_files:
        with open(target, 'r') as f:
            lines = pd.read_table(target, names=('attention', 'word'))
            file_contents.append(lines)
    # 読み込んだcsvを連結
    concat_df = pd.concat(file_contents)
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
    attn_mean_by_words_df.to_csv("./attn_mean_by_words_df.csv")
