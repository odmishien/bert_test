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
    seaborn.histplot(data=df, x='attention', hue='word').set_title(title)
    plt.xlim(0, 1)
    plt.show()


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
    # attention高そうなやつだけ見る
    filtered_df = concat_df.query(f'word == "{args.word}"')
    if len(filtered_df) > 0:
        print("2. plotting...")
        plot(filtered_df, f'attention of {args.word} ({args.target_lp})')
    else:
        print(f'{args.word} does not exist in vocab')
