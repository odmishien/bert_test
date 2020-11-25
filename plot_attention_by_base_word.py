import argparse
import glob
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import japanize_matplotlib


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_lp", type=str,
                        help="Target label and prediction pair.")
    parser.add_argument("--base_word", type=str,
                        help="base word")
    parser.add_argument("--target_dir", type=str, help="Target directory.")
    parser.add_argument("--word", type=str, help="Word")
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


def plot(df, title):
    seaborn.barplot(data=df, x='word', y='attn_mean',
                    order=df.sort_values('attn_mean', ascending=False)['word'][:50]).set_title(title)
    plt.show()


if __name__ == "__main__":
    args = parse_arg()
    print("1. get target files")
    target_files = get_target_files(
        args.target_dir, args.target_attn_layer, args.target_lp)
    print(f'{len(target_files)} files found.')
    print("2. check include base_word")
    file_contents = []
    for target in target_files:
        with open(target, 'r') as f:
            lines = pd.read_table(target, names=('attention', 'word'))
            try:
                if float(lines[lines['word'] == args.base_word]['attention']) > 0.1:
                    file_contents.append(lines)
            except:
                continue
    # 読み込んだcsvを連結
    print(f'{len(file_contents)} files matched.')
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
    print("2. plotting")
    plot_layer = args.target_attn_layer
    if plot_layer != 'all':
        plot_layer = int(plot_layer) + 1
    plot(attn_mean_by_words_df.query('counts > 10'),
         f'attn_mean_with_{args.base_word}(LP:{args.target_lp}/layer:{plot_layer})')
