"""
Adjusted filter indexing for ResNet with proper layer mapping
"""

import json
import math
import argparse
import os

# Configuration
JSON_RANK_PATH = os.path.join('Content_(Raw_Ranks_of_DNN_Model)/RES50', 'filter_ranks_80%_prune.json')
IMPORTANCE = 0.10
OUTPUT_PATH = 'content_for_TMR/ResNet/filter_indices_10pct.json'
BITS_PER_WEIGHT = 31
KERNEL_ELEMS = 3 * 3  # For 3x3 conv kernels
BN_PARAM_MULT = 1     # Count BatchNorm γ parameters

def build_resnet_architecture():
    """Return all 53 convolutional layers with proper block grouping"""
    return [
        # Block 1: Initial layers
        ('Layer_1_Conv2d_3to64', 3, 64, 1),  # conv1
        
        # Block 2: layer1 bottlenecks (3 bottlenecks)
        # Bottleneck 0
        ('Layer_2_Conv2d_64to64', 64, 64, 2),    # conv1
        ('Layer_3_Conv2d_64to64', 64, 64, 2),    # conv2
        ('Layer_4_Conv2d_64to256', 64, 256, 2),  # conv3
        ('Layer_5_Conv2d_64to256', 64, 256, 2),  # downsample
        
        # Bottleneck 1
        ('Layer_6_Conv2d_256to64', 256, 64, 2),
        ('Layer_7_Conv2d_64to64', 64, 64, 2),
        ('Layer_8_Conv2d_64to256', 64, 256, 2),
        
        # Bottleneck 2
        ('Layer_9_Conv2d_256to64', 256, 64, 2),
        ('Layer_10_Conv2d_64to64', 64, 64, 2),
        ('Layer_11_Conv2d_64to256', 64, 256, 2),
        
        # Block 3: layer2 bottlenecks (4 bottlenecks)
        # Bottleneck 0
        ('Layer_12_Conv2d_256to128', 256, 128, 3),
        ('Layer_13_Conv2d_128to128', 128, 128, 3),
        ('Layer_14_Conv2d_128to512', 128, 512, 3),
        ('Layer_15_Conv2d_256to512', 256, 512, 3),  # downsample
        
        # Bottleneck 1
        ('Layer_16_Conv2d_512to128', 512, 128, 3),
        ('Layer_17_Conv2d_128to128', 128, 128, 3),
        ('Layer_18_Conv2d_128to512', 128, 512, 3),
        
        # Bottleneck 2
        ('Layer_19_Conv2d_512to128', 512, 128, 3),
        ('Layer_20_Conv2d_128to128', 128, 128, 3),
        ('Layer_21_Conv2d_128to512', 128, 512, 3),
        
        # Bottleneck 3
        ('Layer_22_Conv2d_512to128', 512, 128, 3),
        ('Layer_23_Conv2d_128to128', 128, 128, 3),
        ('Layer_24_Conv2d_128to512', 128, 512, 3),
        
        # Block 4: layer3 bottlenecks (6 bottlenecks)
        # Bottleneck 0
        ('Layer_25_Conv2d_512to256', 512, 256, 4),
        ('Layer_26_Conv2d_256to256', 256, 256, 4),
        ('Layer_27_Conv2d_256to1024', 256, 1024, 4),
        ('Layer_28_Conv2d_512to1024', 512, 1024, 4),  # downsample
        
        # Bottleneck 1
        ('Layer_29_Conv2d_1024to256', 1024, 256, 4),
        ('Layer_30_Conv2d_256to256', 256, 256, 4),
        ('Layer_31_Conv2d_256to1024', 256, 1024, 4),
        
        # Bottleneck 2
        ('Layer_32_Conv2d_1024to256', 1024, 256, 4),
        ('Layer_33_Conv2d_256to256', 256, 256, 4),
        ('Layer_34_Conv2d_256to1024', 256, 1024, 4),
        
        # Bottleneck 3
        ('Layer_35_Conv2d_1024to256', 1024, 256, 4),
        ('Layer_36_Conv2d_256to256', 256, 256, 4),
        ('Layer_37_Conv2d_256to1024', 256, 1024, 4),
        
        # Bottleneck 4
        ('Layer_38_Conv2d_1024to256', 1024, 256, 4),
        ('Layer_39_Conv2d_256to256', 256, 256, 4),
        ('Layer_40_Conv2d_256to1024', 256, 1024, 4),
        
        # Bottleneck 5
        ('Layer_41_Conv2d_1024to256', 1024, 256, 4),
        ('Layer_42_Conv2d_256to256', 256, 256, 4),
        ('Layer_43_Conv2d_256to1024', 256, 1024, 4),
        
        # Block 5: layer4 bottlenecks (3 bottlenecks)
        # Bottleneck 0
        ('Layer_44_Conv2d_1024to512', 1024, 512, 5),
        ('Layer_45_Conv2d_512to512', 512, 512, 5),
        ('Layer_46_Conv2d_512to2048', 512, 2048, 5),
        ('Layer_47_Conv2d_1024to2048', 1024, 2048, 5),  # downsample
        
        # Bottleneck 1
        ('Layer_48_Conv2d_2048to512', 2048, 512, 5),
        ('Layer_49_Conv2d_512to512', 512, 512, 5),
        ('Layer_50_Conv2d_512to2048', 512, 2048, 5),
        
        # Bottleneck 2
        ('Layer_51_Conv2d_2048to512', 2048, 512, 5),
        ('Layer_52_Conv2d_512to512', 512, 512, 5),
        ('Layer_53_Conv2d_512to2048', 512, 2048, 5),
    ]

def load_ranks(path: str):
    """Load filter ranks with validation"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Rank file not found at: {path}")
    
    with open(path, 'r') as f:
        ranks = json.load(f)
    
    print(f"Loaded ranks with {len(ranks)} layers")
    return ranks

def top_k_filters(filter_dict: dict, frac: float):
    """Return indices of top-`frac` filters by rank"""
    n_filters = len(filter_dict)
    k = max(1, math.ceil(frac * n_filters))
    ordered = sorted(filter_dict.items(), key=lambda kv: kv[1], reverse=True)
    return [int(kv[0].split('_')[-1]) for kv in ordered[:k]]

def compute_indices(ranks: dict, importance: float):
    """Build bit-index ranges for selected filters"""
    arch = build_resnet_architecture()
    scalar_offset = 0
    result = {}
    matched_layers = 0

    for layer_key, in_c, out_c, block_id in arch:
        if layer_key not in ranks:
            print(f"Warning: Layer {layer_key} not found in rank file - skipping")
            continue
            
        wpf = in_c * KERNEL_ELEMS
        total_layer_weights = wpf * out_c
        filt_indices = top_k_filters(ranks[layer_key], importance)
        
        if not filt_indices:
            print(f"Warning: No filters selected for {layer_key}")
            continue

        matched_layers += 1
        block_key = f'Block_{block_id}'
        result.setdefault(block_key, {}).setdefault(layer_key, {})

        for fidx in filt_indices:
            first_bit = (scalar_offset + fidx * wpf) * BITS_PER_WEIGHT
            last_bit = first_bit + wpf * BITS_PER_WEIGHT - 1
            result[block_key][layer_key][f'Filter_{fidx}'] = [first_bit, last_bit]

        scalar_offset += total_layer_weights + BN_PARAM_MULT * out_c

    print(f"Successfully processed {matched_layers}/53 layers")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--importance', type=float, default=IMPORTANCE)
    parser.add_argument('--json_path', type=str, default=JSON_RANK_PATH)
    parser.add_argument('--output', type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    # Load ranks
    try:
        ranks = load_ranks(args.json_path)
    except Exception as e:
        print(f"Error loading ranks: {e}")
        return

    # Compute indices
    indices_json = compute_indices(ranks, args.importance)
    
    if not indices_json:
        print("Error: No indices generated. Check layer names match between:")
        print("- Your rank JSON file")
        print("- The build_resnet_architecture() function")
        return

    # Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(indices_json, f, indent=2)
    print(f"Successfully wrote indices to {args.output}")

if __name__ == '__main__':
    main()