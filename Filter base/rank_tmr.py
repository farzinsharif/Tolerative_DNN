#!/usr/bin/env python3
# ---------------------------------------------------------------
# Priority-TMR repair with BatchNorm channel copy + BN-recal
# ---------------------------------------------------------------

import json, math, os, torch, torch.nn as nn, torchvision
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader

# ---------------- user knobs ----------------
tmr_percentage   = 0.1          # e.g. 0.30 = top-rank 30 % filters per layer
rank_json_path   = Path("filter_ranks.json")
bn_recal_batches = 50            # number of mini-batches for BN recalibration
# --------------------------------------------

BASELINE_PATH = Path("model/baseline_vgg16.pt")
FAULTY_PATH   = Path("model/faulty_vgg16.pt")
REPAIRED_PATH = Path("model/priority_tmr.pt")

os.makedirs("content", exist_ok=True)
pct_tag   = str(tmr_percentage).replace(".", "p")
AUDIT_JSON = Path(f"content/priority_tmr_{pct_tag}.json")

# ---------- VGG-16 definition ----------
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

# ---------- CIFAR-10 data ----------
tfm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
])
train_loader = DataLoader(
    torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=tfm),
    batch_size=256, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(
    torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tfm),
    batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

def accuracy(net):
    net.eval(); correct=total=0
    with torch.no_grad():
        for x,y in test_loader:
            p = net(x.to(device)).argmax(1)
            correct += (p == y.to(device)).sum().item(); total += y.size(0)
    return 100.*correct/total

# ---------- load checkpoints ----------
baseline = make_vgg16().to(device)
faulty   = make_vgg16().to(device)
baseline.load_state_dict(torch.load(BASELINE_PATH, map_location=device))
faulty.load_state_dict  (torch.load(FAULTY_PATH,   map_location=device))

repaired = make_vgg16().to(device)
repaired.load_state_dict(torch.load(FAULTY_PATH,   map_location=device))

# ---------- read rank JSON ----------
rank_data = json.load(open(rank_json_path))
layer_ranks = [rank_data[k]["avg_ranks"]
               for k in sorted(rank_data.keys(), key=lambda s:int(s.split("_")[1]))]

# ---------- repair loop ----------
audit = defaultdict(dict)
examined=fixed=0; tol=1e-7
base_mods, rep_mods = list(baseline.modules()), list(repaired.modules())
conv_idx = 0

for idx,(b_mod,r_mod) in enumerate(zip(base_mods, rep_mods)):
    if not isinstance(b_mod, nn.Conv2d):
        continue

    ranks = layer_ranks[conv_idx]                 # list of floats
    n_filters = len(ranks)
    k = max(1, math.ceil(n_filters * tmr_percentage))
    # pick HIGHEST-rank filters (larger value = more important)
    top_idx = sorted(range(n_filters), key=lambda i: ranks[i], reverse=True)[:k]

    # find next BN
    bn_b = bn_r = None
    for j in range(idx+1, len(base_mods)):
        if isinstance(base_mods[j], nn.BatchNorm2d):
            bn_b, bn_r = base_mods[j], rep_mods[j]; break

    examined += k
    for fi in top_idx:
        status = "not faulty"
        if not torch.allclose(b_mod.weight[fi], r_mod.weight[fi], atol=tol, rtol=0):
            with torch.no_grad():
                r_mod.weight[fi] = b_mod.weight[fi].clone()
                if b_mod.bias is not None:
                    r_mod.bias[fi] = b_mod.bias[fi].clone()
                if bn_b is not None:
                    bn_r.weight[fi]       = bn_b.weight[fi].clone()
                    bn_r.bias[fi]         = bn_b.bias[fi].clone()
                    bn_r.running_mean[fi] = bn_b.running_mean[fi].clone()
                    bn_r.running_var[fi]  = bn_b.running_var[fi].clone()
            fixed += 1
            status = "faulty"
        audit[f"layer {conv_idx}"][f"filter {fi}"] = status

    conv_idx += 1

# ---------- BatchNorm-recalibration ----------
repaired.train()
with torch.no_grad():
    for i,(x,_) in enumerate(train_loader):
        if i >= bn_recal_batches: break
        _ = repaired(x.to(device))
repaired.eval()

# ---------- save & report ----------
torch.save(repaired.state_dict(), REPAIRED_PATH)

acc_base, acc_faulty, acc_rep = map(accuracy,[baseline, faulty, repaired])

summary = dict(
    baseline_accuracy = round(acc_base,   2),
    faulty_accuracy   = round(acc_faulty, 2),
    repaired_accuracy = round(acc_rep,    2),
    filters_examined  = examined,
    filters_corrected = fixed,
    tmr_percentage    = tmr_percentage
)

js = {"summary": summary}; js.update(audit)
AUDIT_JSON.write_text(json.dumps(js, indent=2))

print("\nAccuracy (%)")
print(f"  Baseline : {acc_base:6.2f}")
print(f"  Faulty   : {acc_faulty:6.2f}")
print(f"  Repaired : {acc_rep:6.2f}\n")
print(f"Filters examined : {examined}")
print(f"Filters corrected: {fixed}")
print(f"\nRepaired model : {REPAIRED_PATH}")
print(f"Audit JSON     : {AUDIT_JSON}")
