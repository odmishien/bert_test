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
    parser.add_argument("--target_dir", type=str, help="Target directory.")
    parser.add_argument("--target_attn_layer", type=str,
                        help="Target Attention Layer.")
    return parser.parse_args()


def get_target_files(target_lp="PP", target_dir="results/masuda/attention_2000", target_attn_layer='0'):
    target_files = glob.glob(
        f'{target_dir}/**/{target_lp}_attn_{target_attn_layer}.tsv', recursive=True)
    return target_files


def plot(target_files):
    file_contents = []
    for target in target_files:
        with open(target, 'r') as f:
            lines = pd.read_table(target, names=('attention', 'word'))
            file_contents.append(lines)
    concat_df = pd.concat(file_contents)
    filtered_df = concat_df.query('attention >= 0.9')
    seaborn.catplot(x="word", data=filtered_df,
                    kind="count", order=filtered_df['word'].value_counts().index)
    plt.show()


if __name__ == "__main__":
    args = parse_arg()
    print("1. get target files")
    target_files = get_target_files(
        args.target_lp, args.target_dir, args.target_attn_layer)
    print(f'{len(target_files)} files found.')
    print("2. plotting...")
    plot(target_files)
