import argparse
import glob
import pandas as pd


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, help="Target directory.")
    parser.add_argument("--target_attn_layer", type=str,
                        help="Target Attention Layer.")
    return parser.parse_args()


def get_target_files(target_dir="results/masuda/attention_2000", target_attn_layer='0'):
    pos_target_files = glob.glob(
        f'{target_dir}/**/PP_attn_{target_attn_layer}.tsv', recursive=True)
    nega_target_files = glob.glob(
        f'{target_dir}/**/NN_attn_{target_attn_layer}.tsv', recursive=True)
    return pos_target_files, nega_target_files


def get_high_attention_words(target_files):
    file_contents = []
    for target in target_files:
        with open(target, 'r') as f:
            lines = pd.read_table(target, names=('attention', 'word'))
            file_contents.append(lines)
    # 読み込んだcsvを連結
    concat_df = pd.concat(file_contents)
    # attention高そうなやつだけ見る
    filtered_df = concat_df.query('attention >= 0.9')
    return filtered_df['word'].values.tolist()


def get_unique_value(pos_words, nega_words):
    intersection = set(pos_words) & set(nega_words)
    return list(intersection ^ set(pos_words)), list(intersection ^ set(nega_words))


if __name__ == "__main__":
    args = parse_arg()
    pos, nega = get_target_files(
        args.target_dir, args.target_attn_layer)
    pos_words = get_high_attention_words(pos)
    nega_words = get_high_attention_words(nega)
    pos_unique, nega_unique = get_unique_value(pos_words, nega_words)
    print('pos_unique:')
    print(pos_unique)

    print('nega_unique:')
    print(nega_unique)
