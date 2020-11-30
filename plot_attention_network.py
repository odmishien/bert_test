import pandas as pd
import glob
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import japanize_matplotlib
from tqdm import tqdm


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


def plot(df):
    G = nx.from_pandas_edgelist(df, edge_attr=True)
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(G,
                     node_shape="s",
                     node_color="c",
                     node_size=500,
                     edge_color="gray",
                     font_family="IPAexGothic")
    plt.show()


if __name__ == "__main__":
    args = parse_arg()
    print("1. get target files")
    target_files = get_target_files(
        args.target_dir, args.target_attn_layer, args.target_lp)
    print(f'{len(target_files)} files found.')

    print("2. get base words")
    target_df = pd.read_csv(
        f'./results/attn_mean_20201128_by_words_{args.target_lp}_{args.target_attn_layer}.csv')
    target_df = target_df.query('counts > 500')
    # [UNK] と [CLS]は削除
    target_df = target_df.query('word != "[UNK]"')
    target_df = target_df.query('word != "[CLS]"')
    # attentionの高い上位30語をbaseにする
    base_words = target_df.sort_values(
        'attn_mean', ascending=False)['word'][:30]
    print(base_words)

    temp_attn = []
    plot_df = pd.DataFrame(columns=['source', 'target', 'weight'])
    for bw in tqdm(base_words):
        for target in target_files:
            with open(target, 'r') as f:
                lines = pd.read_table(target, names=('attention', 'word'))
                try:
                    if not lines[lines['word'] == bw].empty:
                        temp_attn.append(lines)
                except:
                    continue
        temp_concat_df = pd.concat(temp_attn)
        temp_plot_df = pd.DataFrame(columns=['source', 'target', 'weight'])
        uniq_words = temp_concat_df.sort_values(
            'attention', ascending=False).word.unique()[:500]
        for w in uniq_words:
            temp_filtered_df = temp_concat_df.query(f'word == "{w}"')
            record = pd.Series(
                [bw, w, temp_filtered_df['attention'].mean()], index=temp_plot_df.columns)
            temp_plot_df = temp_plot_df.append(record, ignore_index=True)
        temp_plot_df = temp_plot_df.query('weight > 0.5')
        plot_df = pd.concat([plot_df, temp_plot_df])
    plot(plot_df)
