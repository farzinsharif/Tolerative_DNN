import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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
output_dir = "Content_(Raw_Ranks_of_DNN_Model)/RES50"
os.makedirs(output_dir, exist_ok=True)

# List all model files
model_dir = "model/RES50"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]

def get_conv_layers(model):
    """Get all Conv2d layers in the model, including those in Bottleneck blocks"""
    conv_layers = []
    
    # Initial conv layer
    if hasattr(model, 'conv1') and isinstance(model.conv1, nn.Conv2d):
        conv_layers.append(model.conv1)
    
    # Process each layer (layer1, layer2, etc.)
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            for block in layer:
                # Bottleneck blocks have conv1, conv2, conv3
                if hasattr(block, 'conv1') and isinstance(block.conv1, nn.Conv2d):
                    conv_layers.append(block.conv1)
                if hasattr(block, 'conv2') and isinstance(block.conv2, nn.Conv2d):
                    conv_layers.append(block.conv2)
                if hasattr(block, 'conv3') and isinstance(block.conv3, nn.Conv2d):
                    conv_layers.append(block.conv3)
                # Downsample may have a conv layer
                if hasattr(block, 'downsample') and block.downsample is not None:
                    for m in block.downsample:
                        if isinstance(m, nn.Conv2d):
                            conv_layers.append(m)
    
    return conv_layers

def forward_hook(module, input, output):
    """Hook to capture the output of a layer"""
    module.output = output

for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    print(f"\nProcessing model: {model_file}")
    
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

    # Get all Conv2d layers
    conv_modules = get_conv_layers(model)
    
    # Register hooks to capture outputs
    hooks = []
    for conv in conv_modules:
        hooks.append(conv.register_forward_hook(forward_hook))

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
            _ = model(images)  # Forward pass (hooks will capture outputs)
            
            for layer_idx, conv in enumerate(conv_modules):
                if hasattr(conv, 'output'):
                    feature_map = conv.output.cpu().squeeze(0)
                    C = feature_map.shape[0]
                    for f in range(C):
                        fm = feature_map[f]
                        rank = torch.linalg.matrix_rank(fm).item()
                        conv_ranks_sum[layer_idx][f] += rank
            
            count += 1
            if count >= num_images:
                break

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Save results
    results_json = {}
    for layer_idx, conv in enumerate(conv_modules, start=1):
        in_ch = conv.in_channels
        out_ch = conv.out_channels
        ranks = conv_ranks_sum[layer_idx-1]
        avg_ranks = [r / count for r in ranks]

        layer_name = f"Layer_{layer_idx}_Conv2d_{in_ch}to{out_ch}"
        results_json[layer_name] = {f"Filter_{i}": round(r, 3) for i, r in enumerate(avg_ranks)}

    # Save JSON
    with open(os.path.join(output_dir, json_name), "w") as f:
        json.dump(results_json, f, indent=4)

    print(f"Done: saved to {json_name}")