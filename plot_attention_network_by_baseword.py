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
    parser.add_argument("--base_word", type=str,
                        help="base word")
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
    pos = nx.spring_layout(G, k=0.3)
    edge_width = [d['weight'] for (u, v, d) in G.edges(data=True)]
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(G, pos,
                     node_color="c",
                     edge_color='blue',
                     width=edge_width,
                     font_family="IPAexGothic")
    plt.show()


if __name__ == "__main__":
    args = parse_arg()
    print("1. get target files")
    target_files = get_target_files(
        args.target_dir, args.target_attn_layer, args.target_lp)
    print(f'{len(target_files)} files found.')

    print("2. generate network")
    bw = args.base_word
    temp_attn = []
    plot_df = pd.DataFrame(columns=['source', 'target', 'weight'])
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
    first_uniq_words = temp_concat_df.word.unique()
    for w in tqdm(first_uniq_words):
        temp_filtered_df = temp_concat_df.query(f'word == "{w}"')
        record = pd.Series(
            [bw, w, temp_filtered_df['attention'].mean()], index=temp_plot_df.columns)
        temp_plot_df = temp_plot_df.append(record, ignore_index=True)
    temp_plot_df = temp_plot_df.query('weight > 0.4')
    next_words = temp_plot_df.sort_values(
        "weight", ascending=False).target.unique()
    print(next_words)
    for sbw in tqdm(next_words):
        second_temp_attn = []
        for t in target_files:
            with open(t, 'r') as f:
                lines = pd.read_table(t, names=('attention', 'word'))
                try:
                    if not lines[lines['word'] == sbw].empty:
                        second_temp_attn.append(lines)
                except:
                    continue
        second_temp_concat_df = pd.concat(second_temp_attn)
        second_uniq_words = second_temp_concat_df.word.unique()
        for w in tqdm(second_uniq_words):
            temp_filtered_df = temp_concat_df.query(f'word == "{w}"')
            record = pd.Series(
                [sbw, w, temp_filtered_df['attention'].mean()], index=temp_plot_df.columns)
            temp_plot_df = temp_plot_df.append(record, ignore_index=True)
    temp_plot_df = temp_plot_df.query('weight > 0.2')
    plot_df = pd.concat([plot_df, temp_plot_df])
    plot(plot_df)
