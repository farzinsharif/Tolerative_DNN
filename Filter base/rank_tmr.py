#!/usr/bin/env python3
# ---------------------------------------------------------------
# Priority-TMR repair (low-rank filters are most important)
# ---------------------------------------------------------------
import json, math, os, torch, torch.nn as nn, torchvision
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader

# ---------------- user knobs ----------------
tmr_percentage = 0.3         # e.g. 0.10 = top 10 % MOST-important filters
rank_json_path = Path("filter_ranks.json")
# --------------------------------------------

BASELINE_PATH  = Path("model/baseline_vgg16.pt")
FAULTY_PATH    = Path("model/faulty_vgg16.pt")
REPAIRED_PATH  = Path("model/priority_tmr.pt")
os.makedirs("content", exist_ok=True)
pct_tag   = str(tmr_percentage).replace(".", "p")
AUDIT_JSON = Path(f"content/priority_tmr_{pct_tag}.json")

# ---------- network ----------
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

# ---------- CIFAR-10 test set ----------
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
])
test_loader = DataLoader(
    torchvision.datasets.CIFAR10("./data", train=False, download=True,
                                 transform=transform),
    batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

def accuracy(net):
    net.eval(); correct=total=0
    with torch.no_grad():
        for x,y in test_loader:
            out = net(x.to(device)).argmax(1)
            correct += (out==y.to(device)).sum().item(); total+=y.size(0)
    return 100.*correct/total

# ---------- load checkpoints ----------
baseline = make_vgg16().to(device)
faulty   = make_vgg16().to(device)
baseline.load_state_dict(torch.load(BASELINE_PATH ,map_location=device))
faulty.load_state_dict  (torch.load(FAULTY_PATH   ,map_location=device))

repaired = make_vgg16().to(device)
repaired.load_state_dict(torch.load(FAULTY_PATH, map_location=device))

# ---------- read rank json ----------
rank_data = json.load(open(rank_json_path))
layer_ranks = []
for key in sorted(rank_data.keys(), key=lambda x:int(x.split("_")[1])):
    layer_ranks.append(rank_data[key]["avg_ranks"])

# ---------- repair ----------
audit = defaultdict(dict)
examined=fixed=0; tol=1e-7
conv_idx=0
with torch.no_grad():
    for b_layer,r_layer in zip(baseline.modules(),repaired.modules()):
        if not isinstance(b_layer, nn.Conv2d): continue
        ranks = layer_ranks[conv_idx]
        n = len(ranks); k = max(1, math.ceil(n*tmr_percentage))
        # pick LOWEST rank values  ← fixed line
        top = sorted(range(n), key=lambda i: ranks[i])[:k]

        examined += k
        for fi in top:
            status="not faulty"
            if not torch.allclose(b_layer.weight[fi], r_layer.weight[fi],
                                  atol=tol, rtol=0):
                r_layer.weight[fi]=b_layer.weight[fi].clone()
                if b_layer.bias is not None:
                    r_layer.bias[fi]=b_layer.bias[fi].clone()
                status="faulty"; fixed+=1
            audit[f"layer {conv_idx}"][f"filter {fi}"]=status
        conv_idx+=1

# ---------- save & report ----------
torch.save(repaired.state_dict(), REPAIRED_PATH)

acc_base, acc_faulty, acc_rep = map(accuracy,[baseline,faulty,repaired])

summary = dict(baseline_accuracy=round(acc_base,2),
               faulty_accuracy  =round(acc_faulty,2),
               repaired_accuracy=round(acc_rep,2),
               filters_examined =examined,
               filters_corrected=fixed,
               tmr_percentage   =tmr_percentage)
out = {"summary": summary}; out.update(audit)
AUDIT_JSON.write_text(json.dumps(out,indent=2))

print("\nAccuracy (%)")
print(f"  Baseline : {acc_base:6.2f}")
print(f"  Faulty   : {acc_faulty:6.2f}")
print(f"  Repaired : {acc_rep:6.2f}\n")
print(f"Filters examined : {examined}")
print(f"Filters corrected: {fixed}")
print(f"\nRepaired model : {REPAIRED_PATH}")
print(f"Audit JSON     : {AUDIT_JSON}")
