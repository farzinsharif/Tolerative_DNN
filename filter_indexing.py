"""
This code_module prepare the weight index in json format suitable for TMR_IN_CNN.py from Rank_generator folder
"""


import json
import math
import argparse
import os

# ---------------- Configuration ---------------- #
JSON_RANK_PATH = os.path.join('content_for_TMR/80%_Prune', 'filter_ranks_80%_prune.json')       # per‑layer filter‑rank file
IMPORTANCE      = 0.35                                          # top‑k fraction of filters (0–1]
OUTPUT_PATH     = 'content_for_TMR/80%_Prune/filter_indices.json'                 # where to write the result
BITS_PER_WEIGHT = 31                                            # mutable bits per float32
KERNEL_ELEMS    = 3 * 3                                         # kernel size (3×3)
BN_PARAM_MULT   = 1                                             # BatchNorm: count γ only (no β)
# ------------------------------------------------ #


def build_architecture():
    """Return the ordered convolutional layers: (key, in_c, out_c, block_id)."""
    return [
        ('Layer_1_Conv2d_3to64',       3,   64, 1),
        ('Layer_2_Conv2d_64to128',    64,  128, 2),
        ('Layer_3_Conv2d_128to256',  128,  256, 3),
        ('Layer_4_Conv2d_256to256',  256,  256, 3),
        ('Layer_5_Conv2d_256to512',  256,  512, 4),
        ('Layer_6_Conv2d_512to512',  512,  512, 4),
        ('Layer_7_Conv2d_512to512',  512,  512, 5),
        ('Layer_8_Conv2d_512to512',  512,  512, 5),
    ]


def load_ranks(path: str):
    """Load the filter‑rank JSON produced by SVD or another method."""
    with open(path, 'r') as f:
        return json.load(f)


def top_k_filters(filter_dict: dict, frac: float):
    """Return indices (0‑based) of the top‑`frac` filters by rank (descending)."""
    n_filters = len(filter_dict)
    k = max(1, math.ceil(frac * n_filters))
    ordered = sorted(filter_dict.items(), key=lambda kv: kv[1], reverse=True)
    return [int(kv[0].split('_')[-1]) for kv in ordered[:k]]


def compute_indices(ranks: dict, importance: float):
    """Build a nested JSON describing bit‑index ranges for the selected filters."""
    arch = build_architecture()
    scalar_offset = 0  # counts *weights* seen so far (not bits)
    result = {}

    for layer_key, in_c, out_c, block_id in arch:
        wpf = in_c * KERNEL_ELEMS                          # weights per filter
        total_layer_weights = wpf * out_c                  # whole conv layer

        # pick top‑k filters for this layer
        filt_indices = top_k_filters(ranks[layer_key], importance)

        block_key = f'Block_{block_id}'
        result.setdefault(block_key, {}).setdefault(layer_key, {})

        for fidx in filt_indices:
            first_bit = (scalar_offset + fidx * wpf) * BITS_PER_WEIGHT
            last_bit  = first_bit + wpf * BITS_PER_WEIGHT - 1
            result[block_key][layer_key][f'Filter_{fidx}'] = [first_bit, last_bit]

        # advance offset: conv weights + BatchNorm γ (skip β)
        scalar_offset += total_layer_weights + BN_PARAM_MULT * out_c

    return result


def main():
    parser = argparse.ArgumentParser(description='Generate bit‑index ranges for top‑ranked filters.')
    parser.add_argument('--importance', type=float, default=IMPORTANCE,
                        help='Fraction (0‑1] of top filters to keep per layer.')
    parser.add_argument('--json_path', type=str, default=JSON_RANK_PATH,
                        help='Path to the filter‑rank JSON file.')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH,
                        help='Output JSON file name.')
    args = parser.parse_args()
    importance_pct = int(args.importance * 100)
    base, ext = os.path.splitext(args.output)
    args.output = f"{base}_{importance_pct}pct{ext}"

    ranks = load_ranks(args.json_path)
    indices_json = compute_indices(ranks, args.importance)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(indices_json, f, indent=2)
    print(f'Wrote indices to {args.output}')


if __name__ == '__main__':
    main()
