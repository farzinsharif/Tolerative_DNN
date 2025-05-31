#!/usr/bin/env python3
# Random TMR repair for VGG-16 + BN-recalibration
import json, math, os, random, torch, torch.nn as nn, torchvision
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader

# ---------- user knobs ----------
tmr_percentage = 0.10   # fraction of Conv filters to test per layer
random_seed    = 123   # None → fresh randomness; int → repeatable subset
bn_recal_batches = 50   # ▶ how many batches to use for BN-recalibration
# ---------------------------------

BASELINE = Path("model/baseline_vgg16.pt")
FAULTY   = Path("model/faulty_vgg16.pt")
REPAIRED = Path("model/random_tmr.pt")
os.makedirs("content", exist_ok=True)
pct_tag = str(tmr_percentage).replace(".", "p")
AUDIT   = Path(f"content/random_tmr_{pct_tag}.json")

def make_vgg16():
    net = torchvision.models.vgg16_bn(pretrained=False)
    net.features[0] = nn.Conv2d(3,64,kernel_size=3,padding=1)
    net.avgpool = nn.AdaptiveAvgPool2d((1,1))
    net.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512,512), nn.ReLU(True), nn.Dropout(),
        nn.Linear(512,512), nn.ReLU(True), nn.Dropout(),
        nn.Linear(512,10))
    return net

device = "cuda" if torch.cuda.is_available() else "cpu"
if random_seed is not None:
    random.seed(random_seed); torch.manual_seed(random_seed)
else:
    random.seed()

# ---------- CIFAR-10 loaders ----------
tfm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914,0.4822,0.4465),
                                     (0.2023,0.1994,0.2010))])
train_loader = DataLoader(
    torchvision.datasets.CIFAR10("./data",train=True ,download=True,transform=tfm),
    batch_size=256, shuffle=True ,num_workers=2, pin_memory=True)
test_loader  = DataLoader(
    torchvision.datasets.CIFAR10("./data",train=False,download=True,transform=tfm),
    batch_size=256, shuffle=False,num_workers=2, pin_memory=True)

def accuracy(net):
    net.eval(); tot=correct=0
    with torch.no_grad():
        for x,y in test_loader:
            p=net(x.to(device)).argmax(1)
            correct+=(p==y.to(device)).sum().item(); tot+=y.size(0)
    return 100.*correct/tot

# ---------- load nets ----------
baseline = make_vgg16().to(device)
faulty   = make_vgg16().to(device)
baseline.load_state_dict(torch.load(BASELINE,map_location=device))
faulty.load_state_dict  (torch.load(FAULTY  ,map_location=device))
repaired = make_vgg16().to(device)
repaired.load_state_dict(torch.load(FAULTY  ,map_location=device))

base_mod, rep_mod = list(baseline.modules()), list(repaired.modules())

# ---------- TMR filter repair ----------
audit=defaultdict(dict); examined=fixed=0; tol=1e-7; conv_idx=0
for idx,(b,r) in enumerate(zip(base_mod,rep_mod)):
    if not isinstance(b,nn.Conv2d): continue
    n=b.weight.size(0); k=max(1,math.ceil(n*tmr_percentage))
    sel=random.sample(range(n),k); examined+=k

    # find next BN
    bn_b=bn_r=None
    for nxt in range(idx+1,len(base_mod)):
        if isinstance(base_mod[nxt],nn.BatchNorm2d):
            bn_b, bn_r = base_mod[nxt], rep_mod[nxt]; break

    for fi in sel:
        st="not faulty"
        if not torch.allclose(b.weight[fi], r.weight[fi],atol=tol,rtol=0):
            with torch.no_grad():
                r.weight[fi]=b.weight[fi].clone()
                if b.bias is not None: r.bias[fi]=b.bias[fi].clone()
                if bn_b is not None:
                    bn_r.weight[fi]=bn_b.weight[fi].clone()
                    bn_r.bias[fi]  =bn_b.bias[fi].clone()
                    bn_r.running_mean[fi]=bn_b.running_mean[fi].clone()
                    bn_r.running_var [fi]=bn_b.running_var [fi].clone()
            st="faulty"; fixed+=1
        audit[f"layer {conv_idx}"][f"filter {fi}"]=st
    conv_idx+=1

# ▶ ---------- BatchNorm-recalibration ----------
repaired.train()                  # BN updates running stats
with torch.no_grad():
    for i,(x,_) in enumerate(train_loader):
        if i>=bn_recal_batches: break
        _=repaired(x.to(device))
repaired.eval()
# ----------------------------------------------

# ---------- save + report ----------
torch.save(repaired.state_dict(), REPAIRED)
acc_base,acc_fault,acc_rep = map(accuracy,[baseline,faulty,repaired])

summary=dict(baseline_accuracy=round(acc_base,2),
             faulty_accuracy  =round(acc_fault,2),
             repaired_accuracy=round(acc_rep,2),
             filters_examined =examined,
             filters_corrected=fixed,
             tmr_percentage   =tmr_percentage,
             random_seed      =random_seed)
out={"summary":summary}; out.update(audit)
AUDIT.write_text(json.dumps(out,indent=2))

print("\nAccuracy (%)")
print(f"  Baseline : {acc_base:6.2f}")
print(f"  Faulty   : {acc_fault:6.2f}")
print(f"  Repaired : {acc_rep:6.2f}\n")
print(f"Filters examined : {examined}")
print(f"Filters corrected: {fixed}")
print(f"\nRepaired model : {REPAIRED}")
print(f"Audit JSON     : {AUDIT}")
