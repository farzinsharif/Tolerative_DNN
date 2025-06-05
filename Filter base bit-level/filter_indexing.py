import json
import math
import argparse
import os

# ---------------- Configuration ---------------- #
JSON_RANK_PATH = os.path.join('content', 'svd_rank.json')
IMPORTANCE = 0.1  # Top 10 % of filters per layer (0–1, inclusive)
OUTPUT_PATH = 'content/filter_indices.json'
BITS_PER_WEIGHT = 31
KERNEL_ELEMS = 3 * 3
# ------------------------------------------------ #

def build_architecture():
    """Returns ordered list of convolutional layers with metadata."""
    return [
        # (layer_key, in_channels, out_channels, block_id)
        ('Layer_1_Conv2d_3to64',   3,   64, 1),
        ('Layer_2_Conv2d_64to128', 64,  128, 2),
        ('Layer_3_Conv2d_128to256',128, 256, 3),
        ('Layer_4_Conv2d_256to256',256, 256, 3),
        ('Layer_5_Conv2d_256to512',256, 512, 4),
        ('Layer_6_Conv2d_512to512',512, 512, 4),
        ('Layer_7_Conv2d_512to512',512, 512, 5),
        ('Layer_8_Conv2d_512to512',512, 512, 5),
    ]


def load_ranks(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def top_k_filters(filter_dict: dict, frac: float):
    """Return indices of top-`frac` filters by rank descending."""
    n_filters = len(filter_dict)
    k = max(1, math.ceil(frac * n_filters))
    # sort descending by rank value
    ordered = sorted(filter_dict.items(), key=lambda x: x[1], reverse=True)
    return [int(kv[0].split('_')[-1]) for kv in ordered[:k]]


def compute_indices(ranks: dict, importance: float):
    arch = build_architecture()
    scalar_offset = 0  # counts weights (not bits) already passed
    result = {}

    for layer_key, in_c, out_c, block_id in arch:
        wpf = in_c * KERNEL_ELEMS                   # weights per filter
        total_layer_weights = wpf * out_c
        block_key = f'Block_{block_id}'

        # get top filters in this layer
        filt_indices = top_k_filters(ranks[layer_key], importance)

        # initialise nested dicts
        result.setdefault(block_key, {}).setdefault(layer_key, {})

        for fidx in filt_indices:
            first_bit = (scalar_offset + fidx * wpf) * BITS_PER_WEIGHT
            last_bit  = first_bit + wpf * BITS_PER_WEIGHT - 1
            result[block_key][layer_key][f'Filter_{fidx}'] = [first_bit, last_bit]

        scalar_offset += total_layer_weights

    return result


def main():
    parser = argparse.ArgumentParser(description='Compute index ranges for top-ranked filters.')
    parser.add_argument('--importance', type=float, default=IMPORTANCE,
                        help='Fraction (0-1) of top filters to keep per layer.')
    parser.add_argument('--json_path', type=str, default=JSON_RANK_PATH,
                        help='Path to JSON file containing filter ranks.')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH,
                        help='Output JSON file name.')
    args = parser.parse_args()

    ranks = load_ranks(args.json_path)
    indices_json = compute_indices(ranks, args.importance)

    with open(args.output, 'w') as f:
        json.dump(indices_json, f, indent=2)
    print(f'Wrote indices to {args.output}')


if __name__ == '__main__':
    main()
