#!/usr/bin/env python3
# ---------------------------------------------------------------
# Random TMR spot-check & repair for VGG-16 filters
# ---------------------------------------------------------------
# • Samples tmr_percentage of filters per Conv layer
# • Compares faulty vs. baseline; copies baseline weights/biases
#   into the faulty model where mismatches are found
# • Saves:
#     - model/random_tmr.pt                (repaired checkpoint)
#     - content/random_tmr_<pct>.json      (audit + summary)
# • Prints accuracy for baseline, faulty, repaired
# ---------------------------------------------------------------

import os, json, math, random, torch, torchvision, torch.nn as nn
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader

# ----------------------- user parameters -----------------------
tmr_percentage = 0.10   # 0.10 → check 10 % of filters in each Conv layer
random_seed    = 123   # None = fresh random every run; set int for repeatability
# ---------------------------------------------------------------

BASELINE_PATH = Path("model/baseline_vgg16.pt")
FAULTY_PATH   = Path("model/faulty_vgg16.pt")
REPAIRED_PATH = Path("model/random_tmr.pt")

os.makedirs("content", exist_ok=True)
pct_tag   = str(tmr_percentage).replace(".", "p")
JSON_PATH = Path(f"content/random_tmr_{pct_tag}.json")

# ----------------------- network definition -------------------
def make_vgg16():
    net = torchvision.models.vgg16_bn(pretrained=False)
    net.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    net.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(),
        nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(),
        nn.Linear(512, 10),
    )
    return net

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------- seeding -----------------------------
if random_seed is not None:
    random.seed(random_seed)
    torch.manual_seed(random_seed)
else:
    random.seed()           # time / OS entropy
# ---------------------------------------------------------------

# -------------------------- dataset ---------------------------
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
])
testset  = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(testset, batch_size=256,
                         shuffle=False, num_workers=2, pin_memory=True)

def accuracy(net):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = net(imgs).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total

# ---------------------- load the models -----------------------
baseline = make_vgg16().to(device)
faulty   = make_vgg16().to(device)
baseline.load_state_dict(torch.load(BASELINE_PATH, map_location=device))
faulty.load_state_dict(torch.load(FAULTY_PATH,   map_location=device))

repaired = make_vgg16().to(device)
repaired.load_state_dict(torch.load(FAULTY_PATH, map_location=device))

# ------------------ spot-check and repair ---------------------
audit_log = defaultdict(dict)        # { "layer N": { "filter i": "faulty/ok" } }
sampled_total = 0
fixed_total   = 0
tol = 1e-7

with torch.no_grad():
    for layer_idx, (b_layer, r_layer) in enumerate(
            zip(baseline.modules(), repaired.modules())):
        if not isinstance(b_layer, nn.Conv2d):
            continue

        n_filters = b_layer.weight.size(0)
        n_sample  = max(1, int(math.ceil(n_filters * tmr_percentage)))
        indices   = random.sample(range(n_filters), n_sample)
        sampled_total += n_sample

        for fi in indices:
            layer_key  = f"layer {layer_idx}"
            filter_key = f"filter {fi}"
            if not torch.allclose(b_layer.weight[fi],
                                  r_layer.weight[fi], atol=tol, rtol=0):
                r_layer.weight[fi] = b_layer.weight[fi].clone()
                if b_layer.bias is not None:
                    r_layer.bias[fi] = b_layer.bias[fi].clone()
                audit_log[layer_key][filter_key] = "faulty"
                fixed_total += 1
            else:
                audit_log[layer_key][filter_key] = "not faulty"

# --------------------------- save ------------------------------
torch.save(repaired.state_dict(), REPAIRED_PATH)

acc_baseline = accuracy(baseline)
acc_faulty   = accuracy(faulty)
acc_repaired = accuracy(repaired)

summary = {
    "baseline_accuracy": round(acc_baseline, 2),
    "faulty_accuracy"  : round(acc_faulty, 2),
    "repaired_accuracy": round(acc_repaired, 2),
    "filters_examined" : sampled_total,
    "filters_corrected": fixed_total,
    "tmr_percentage"   : tmr_percentage,
    "random_seed"      : random_seed
}

json_data = {"summary": summary}
json_data.update(audit_log)
JSON_PATH.write_text(json.dumps(json_data, indent=2))

# -------------------------- console ---------------------------
print("\nAccuracy (%)")
print(f"  Baseline : {acc_baseline:6.2f}")
print(f"  Faulty   : {acc_faulty:6.2f}")
print(f"  Repaired : {acc_repaired:6.2f}\n")

print(f"Filters examined : {sampled_total}")
print(f"Filters corrected: {fixed_total}")
print(f"\nRepaired model saved to : {REPAIRED_PATH}")
print(f"Audit JSON saved to     : {JSON_PATH}")
