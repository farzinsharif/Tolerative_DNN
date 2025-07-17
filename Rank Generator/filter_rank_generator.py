import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# import seaborn as sns
import json
import os
import re

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset: CIFAR-100
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])
dataset = datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Output directory
output_dir = "Content"
os.makedirs(output_dir, exist_ok=True)

# List all model files
model_dir = "model/VGG11"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]

for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    print(f"\n Processing model: {model_file}")
    
    # Load model
    model = torch.load(model_path, map_location=device)
    model.eval()

    # Get pruning percentage for filename
    match = re.search(r'_(\d+)\.pt$', model_file)
    if match:
        prune_pct = match.group(1)
        json_name = f"filter_ranks_{prune_pct}%_prune.json"
    else:
        json_name = "filter_ranks.json"

    # Identify Conv2d layers
    conv_indices = [i for i, layer in enumerate(model.features) if isinstance(layer, nn.Conv2d)]
    conv_modules = [model.features[i] for i in conv_indices]

    # Init rank accumulators
    conv_ranks_sum = []
    for conv in conv_modules:
        conv_ranks_sum.append([0.0] * conv.out_channels)

    # Process images
    num_images = 100
    count = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            x = images
            conv_outputs = []
            for layer in model.features:
                x = layer(x)
                if isinstance(layer, nn.Conv2d):
                    conv_outputs.append(x)
            for layer_idx, feature_map in enumerate(conv_outputs):
                feature_map = feature_map.cpu().squeeze(0)
                C = feature_map.shape[0]
                for f in range(C):
                    fm = feature_map[f]
                    rank = torch.linalg.matrix_rank(fm).item()
                    conv_ranks_sum[layer_idx][f] += rank
            count += 1
            if count >= num_images:
                break

    # Save results
    results_json = {}
    for layer_idx, conv in enumerate(conv_modules, start=1):
        in_ch = conv.in_channels
        out_ch = conv.out_channels
        ranks = conv_ranks_sum[layer_idx-1]
        avg_ranks = [r / count for r in ranks]

        layer_name = f"Layer_{layer_idx}_Conv2d_{in_ch}to{out_ch}"
        results_json[layer_name] = {f"Filter_{i}": round(r, 3) for i, r in enumerate(avg_ranks)}

        # # Plot heatmap
        # plt.figure(figsize=(max(6, len(avg_ranks) // 4), 2))
        # sns.heatmap([avg_ranks], cmap="viridis", cbar=True, xticklabels=True, yticklabels=False)
        # plt.title(f"Avg Rank Heatmap: {layer_name}")
        # plt.xlabel("Filter Index")
        # plt.tight_layout()
        # plt.savefig(os.path.join(output_dir, f"{model_file[:-3]}_{layer_name}_heatmap.png"))
        # plt.close()

    # Save JSON
    with open(os.path.join(output_dir, json_name), "w") as f:
        json.dump(results_json, f, indent=4)

    print(f" Done: saved to {json_name}")
