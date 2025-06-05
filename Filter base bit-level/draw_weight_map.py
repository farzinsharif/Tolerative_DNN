import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ───────────── USER CONFIG ───────────── #
IMPORTANCE      = 0.10                                 # e.g., 0.10 → top 10%
TOTAL_INDICES   = 129_167_744                          # fixed grand total
JSON_PATH       = os.path.join('content', 'filter_indices.json')
SAVE_PATH       = f"content/weight chart-{int(IMPORTANCE * 100)}% importance.png"
BITS_PER_WEIGHT = 31
# ─────────────────────────────────────── #

# 1. Load important bit-index ranges
with open(JSON_PATH, "r") as f:
    data = json.load(f)

important_bit_ranges = []

def collect_ranges(node):
    if isinstance(node, list) and len(node) == 2 and all(isinstance(x, int) for x in node):
        important_bit_ranges.append(node)
    elif isinstance(node, dict):
        for v in node.values():
            collect_ranges(v)

collect_ranges(data)

# Convert bit-index → weight-index ranges
important_weight_ranges = [
    (rng[0] // BITS_PER_WEIGHT, rng[1] // BITS_PER_WEIGHT) for rng in important_bit_ranges
]

# 2. Compute BN and FC weight-index ranges
conv_layer_weights = [
    64 * 3 * 3 * 3,         # Conv(3→64)
    128 * 64 * 3 * 3,       # Conv(64→128)
    256 * 128 * 3 * 3,      # Conv(128→256)
    256 * 256 * 3 * 3,      # Conv(256→256)
    512 * 256 * 3 * 3,      # Conv(256→512)
    512 * 512 * 3 * 3,      # Conv(512→512)
    512 * 512 * 3 * 3,      # Conv(512→512)
    512 * 512 * 3 * 3       # Conv(512→512)
]
bn_counts = [64, 128, 256, 256, 512, 512, 512, 512]
fc_counts = [25088 * 4096, 4096 * 4096, 4096 * 100]

current_widx = 0
bn_weight_ranges = []
for conv_w, bn_c in zip(conv_layer_weights, bn_counts):
    current_widx += conv_w
    bn_weight_ranges.append((current_widx, current_widx + bn_c - 1))
    current_widx += bn_c

fc_weight_ranges = []
for cnt in fc_counts:
    fc_weight_ranges.append((current_widx, current_widx + cnt - 1))
    current_widx += cnt

# 3. Plot full horizontal bar
fig = plt.figure(figsize=(16, 4))
ax = fig.add_axes([0.05, 0.45, 0.9, 0.4])

# Base bar: conv weights
ax.barh(0, TOTAL_INDICES, left=0, height=0.7, color="blue")

# BN: green
for start, end in bn_weight_ranges:
    ax.barh(0, end - start + 1, left=start, height=0.7, color="green")

# FC: yellow
for start, end in fc_weight_ranges:
    ax.barh(0, end - start + 1, left=start, height=0.7, color="yellow")

# Important conv filters: red
for start, end in important_weight_ranges:
    if start > TOTAL_INDICES:
        continue
    end = min(end, TOTAL_INDICES - 1)
    ax.barh(0, end - start + 1, left=start, height=0.7, color="red")

# Axis config
ax.set_xlim(0, TOTAL_INDICES)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([])
ax.set_xlabel("Weight index (flattened order)")
ax.set_title(
    f"Weight-index map – red = top {int(IMPORTANCE * 100)}% conv filters",
    pad=20
)

# Legend
ax.legend(
    handles=[
        Patch(facecolor="blue",   label="Conv layer weights"),
        Patch(facecolor="red",    label=f"Important conv weights (top {IMPORTANCE:.0%})"),
        Patch(facecolor="green",  label="Batch-Norm weights"),
        Patch(facecolor="yellow", label="FC-layer weights"),
    ],
    bbox_to_anchor=(0.5, -0.6),
    loc="upper center",
    ncol=2
)

plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
plt.close()

print(f" Saved chart to: {SAVE_PATH}")
