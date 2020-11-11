import argparse
import glob
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import japanize_matplotlib


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, help="Target directory.")
    parser.add_argument("--word", type=str, help="Word")
    parser.add_argument("--target_attn_layer", type=str,
                        help="Target Attention Layer.")
    return parser.parse_args()


def get_target_files(target_dir="results/masuda/attention_2000", target_attn_layer='0'):
    target_files = glob.glob(
        f'{target_dir}/**/**_attn_{target_attn_layer}.tsv', recursive=True)
    return target_files


def plot(df):
    seaborn.histplot(data=df, x='attention', hue='word')
    plt.xlim(0, 1)
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
    # attention高そうなやつだけ見る
    filtered_df = concat_df.query(f'word == "{args.word}"')
    if len(filtered_df) > 0:
        print("2. plotting...")
        plot(filtered_df)
    else:
        print(f'{args.word} does not exist in vocab')
