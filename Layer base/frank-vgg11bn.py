import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Configuration
config = {
    'data_path': './data',
    'job_dir': './rank_results_vgg11_bn',
    'batch_size': 128,
    'limit': 5,
    'gpu': '0'
}

# Set up device
os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preparation (CIFAR-10)
print("==> Preparing CIFAR-10 data...")
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fit ImageNet-trained model
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_data = datasets.CIFAR10(root=config['data_path'], train=True, download=True, transform=train_transform)
train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=2)

# Load torchvision VGG11_bn
print("==> Loading VGG11_bn from torchvision...")
model = models.vgg11_bn(pretrained=True)
model = model.to(device)
model.eval()

# Dictionary to collect ranks
layer_ranks = {}

# Hook function
def get_feature_map_rank(name):
    def hook(module, input, output):
        batch_size = output.shape[0]
        if output.ndim == 4:  # Conv2D: (B, C, H, W)
            C = output.shape[1]
            rank_sum = 0
            for i in range(batch_size):
                for c in range(C):
                    fm = output[i, c].detach().cpu()
                    rank_sum += torch.linalg.matrix_rank(fm).item()
            avg_rank = rank_sum / (batch_size * C)
        elif output.ndim == 2:  # Linear: (B, N)
            rank_sum = 0
            for i in range(batch_size):
                vec = output[i].unsqueeze(0).detach().cpu()
                rank_sum += torch.linalg.matrix_rank(vec).item()
            avg_rank = rank_sum / batch_size
        else:
            avg_rank = 0

        if name not in layer_ranks:
            layer_ranks[name] = []
        layer_ranks[name].append(avg_rank)
    return hook

# Register hooks on Conv and Linear layers
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        module.register_forward_hook(get_feature_map_rank(name))

# Inference and rank calculation
def calculate_ranks():
    print("==> Calculating ranks...")
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(train_loader):
            if batch_idx >= config['limit']:
                break
            inputs = inputs.to(device)
            model(inputs)
            print(f"Processed batch {batch_idx + 1}/{config['limit']}")

    # Average ranks
    avg_layer_ranks = {name: sum(ranks)/len(ranks) for name, ranks in layer_ranks.items()}

    # Save
    os.makedirs(config['job_dir'], exist_ok=True)
    save_path = os.path.join(config['job_dir'], 'vgg11_bn_ranks.json')
    with open(save_path, 'w') as f:
        json.dump(avg_layer_ranks, f, indent=4)

    print(f"==> Layer ranks saved to {save_path}")
    for name, rank in avg_layer_ranks.items():
        print(f"{name}: {rank:.2f}")

# Run
calculate_ranks()
