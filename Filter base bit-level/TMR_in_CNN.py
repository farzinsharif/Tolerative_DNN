#Code Block 1
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import random
import pandas as pd
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
#training data
train_data = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)

#test data
test_data = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)
# Code Block 2 
model_save_name = 'VGG11.pt'
path = F"./model/VGG11/{model_save_name}"
AlexNet_model = torchvision.models.vgg11_bn(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AlexNet_model.to(device)
model = torch.load(path, map_location=device)

model.eval()
# Code Block 3

from ignite.metrics import Precision,Recall,Accuracy,ConfusionMatrix,TopKCategoricalAccuracy
precision = Precision(device=device)
#confusionMatrix=ConfusionMatrix(10,device=device)
recall=Recall(device=device)
acc=Accuracy(device=device)
T_acc=TopKCategoricalAccuracy(k=5,device=device)
criterion = torch.nn.CrossEntropyLoss()

sm = torch.nn.Softmax(dim=1)
def test(model):
    confi=0
    sub_confi=0
    correct=0
    total=0
    with torch.no_grad():

        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            #loss = criterion(outputs, labels)
            precision.update((outputs, labels))
            #confusionMatrix.update((outputs, labels))
            recall.update((outputs, labels))
            acc.update((outputs, labels))
            T_acc.update((outputs, labels))
            probabilities = sm(outputs)
            topk=torch.topk(probabilities, 1)
            topk2=torch.topk(probabilities, 2)
            cols = torch.chunk(topk2.values, 2, 1)
            sub_confi+=(cols[0].sum()-cols[1].sum())/32
            confi+=topk.values.sum()/32
            index_conf=torch.nonzero(topk.values.reshape(-1)>0.50)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted[index_conf] == labels[index_conf]).sum()
            #print(topk.values)
            total += labels.size(0)

        return_acc=acc.compute()
        return_pre=precision.compute()
        return_rec=recall.compute()
        return_tacc= T_acc.compute()
        return_con=confi/len(testloader)
        return_sub_con=sub_confi/len(testloader)
        precision.reset()
        recall.reset()
        acc.reset()
        #confusionMatrix.reset()
        T_acc.reset()
        acc_50=correct/total
        return return_acc,return_pre,return_rec,return_tacc,return_con.item(),return_sub_con.item(),acc_50.item()
# Code Block 4

# ================================================================
# change number to bit representation
# ================================================================
import numpy as np

def IEEE754_v2_tensor(numbers):

    signs = np.where(numbers < 0, 1, 0)
    numbers = np.abs(numbers)

    int_parts = np.floor(numbers).astype(int)
    dec_parts = numbers - int_parts

    int_bin_parts = np.array([bin(x).replace('0b', '') if x > 0 else '' for x in int_parts])

    mantissas = []
    exponents = []

    for i in range(len(numbers)):
        if int_parts[i] > 0:
            mantissa = int_bin_parts[i][1:] + fractional_to_bin(dec_parts[i], 23 - len(int_bin_parts[i][1:]))
            exponent = len(int_bin_parts[i]) - 1
        else:
            fraction_bin = fractional_to_bin(dec_parts[i], 50)
            first_one = fraction_bin.find('1')
            exponent = -(first_one + 1)
            mantissa = fraction_bin[first_one + 1:first_one + 24]

        mantissa = (mantissa + '0' * 23)[:23]
        mantissas.append(mantissa)
        exponents.append(exponent)

    exponents = np.array(exponents) + 127
    exponent_bits = np.array([bin(e).replace('0b', '').zfill(8) for e in exponents])

    ieee754_representations = np.array([
        str(signs[i]) + exponent_bits[i] + mantissas[i] for i in range(len(numbers))
    ])

    return ieee754_representations


def fractional_to_bin(dec_part, length=24):

    mantissa = ''
    for _ in range(length):
        dec_part *= 2
        int_part = int(dec_part)
        mantissa += str(int_part)
        dec_part -= int_part
        if dec_part == 0:
            break
    return mantissa + '0' * (length - len(mantissa))  # Pad to ensure fixed length

import numpy as np

def inv_IEEE754_tensor(num_IEEE_array):

    binary_matrix = np.array([list(num) for num in num_IEEE_array], dtype=int)
    signs = binary_matrix[:, 0]
    exponent_bits = binary_matrix[:, 1:9]
    exponents = np.dot(exponent_bits, 2 ** np.arange(7, -1, -1))
    mantissa_bits = binary_matrix[:, 9:].astype(float)
    powers = 2.0 ** np.arange(-1, -mantissa_bits.shape[1] - 1, -1, dtype=float)
    mantissas = np.dot(mantissa_bits, powers)
    normalized_mantissas = 1.0 + mantissas
    is_subnormal = (exponents == 0)
    exponents = np.where(is_subnormal, -126, exponents - 127)
    mantissas = np.where(is_subnormal, mantissas, normalized_mantissas)
    is_zero = (exponents == -127) & (mantissa_bits.sum(axis=1) == 0)
    numbers = mantissas * (2.0 ** exponents)
    numbers = np.where(is_zero, 0.0, numbers)
    numbers = np.where(signs == 1, -numbers, numbers)

    return numbers
# Code Block 5
import numpy as np
import torch

def bitFLIP_v3_tensor(original_values, positions_list):
    original_values_np = original_values.cpu().detach().numpy()  # Convert Torch tensor to NumPy array

    ieee_binary_strings = IEEE754_v2_tensor(original_values_np)  # Convert to IEEE 754 binary
    flipped_binaries = []
    for i, positions in enumerate(positions_list):
        str_num = list(ieee_binary_strings[i])
        for position in positions:
            bit_position = 31 - position  # Convert to IEEE754 bit position
            if bit_position == 1:  # Prevent flipping sign bit
                bit_position = 0
            #print(len(str_num))
            str_num[bit_position] = '0' if str_num[bit_position] == '1' else '1'
        if(original_values_np[i]==0):
          #print(original_values_np[i])
          str_num='00000000000000000000000000000000'
        flipped_binaries.append("".join(str_num))

    flipped_values = inv_IEEE754_tensor(np.array(flipped_binaries))

    flipped_values_tensor = torch.tensor(flipped_values, dtype=original_values.dtype, device=original_values.device, requires_grad=True)

    return flipped_values_tensor
# Code Block 6.4 (there were 4 code block in the tmr fun which eventually the 4th code block was the correct one.)
"""
CODE BLOCK 4
"""
import torch
import random
import pandas as pd
from collections import Counter
import json
# BER be soorate block akahar neveshte shode
power = -5
BER = 5 * (10 ** power)

def fault_tolerance_two_agree(BER, model):
    t = torch.cat([param.view(-1) for name, param in model.named_parameters()
                   if "weight" in name and "norm" not in name]).to('cuda:0')

    count = len(t)
    nums = int(count * 31 * BER)
    # print(f"Total number of weights (float32): {count}")

    def random_bit_positions():
        return random.sample(range(0, 31 * count), nums)

    pos1 = random_bit_positions()
    pos2 = random_bit_positions()
    pos3 = random_bit_positions()
    # Optional print statements — comment these out if not needed
    # print(len(pos1))
    # print(len(pos2))
    # print(len(pos3))
    # print(len(pos1 + pos2 + pos3))  # or print(len(all_positions)) after definition
    all_positions = pos1 + pos2 + pos3
    pos_counts = Counter(all_positions)

    # Load valid ranges from the JSON file
    with open('content/filter_indices_60pct.json') as f:
        range_data = json.load(f)

    # Flatten and collect all valid index ranges
    valid_ranges = []
    for block in range_data.values():
        for layer in block.values():
            for bounds in layer.values():
                valid_ranges.append(bounds)

    # Display valid ranges as a DataFrame (optional)
    # display(pd.DataFrame(valid_ranges, columns=["start", "end"]))  # comment this out to suppress output # Note: since 
    # we are converting this py from an ipynb this line of code is only for ipynb represnation i added the below print
    #  line of code instead
    # print(pd.DataFrame(valid_ranges, columns=["start", "end"])) # to see the above command in .py format uncomment this line

    def is_in_valid_range(index):
        return any(start <= index <= end for start, end in valid_ranges)

    # Debug counts after function is defined
    count_repeat = sum(1 for pos, cnt in pos_counts.items() if cnt >= 2)
    count_valid = sum(1 for pos in pos_counts if is_in_valid_range(pos // 31))
    count_both = sum(1 for pos, cnt in pos_counts.items() if cnt >= 2 and is_in_valid_range(pos // 31))

    print(f"Positions repeated >=2 times: {count_repeat}")
    print(f"Positions inside valid range: {count_valid}")
    print(f"Positions satisfying both: {count_both}")

    # Apply both conditions
    two_or_more = [pos for pos, cnt in pos_counts.items()
                   if cnt >= 2 and is_in_valid_range(pos // 31)] #Only implement tmr if in the certain values
    random_pos_in_pos1 = [x for x in pos1 if not is_in_valid_range(x // 31)] # Not TMR fault injection
 
    all_pos = two_or_more + random_pos_in_pos1
    # print(len(all_pos))
    # print(len(two_or_more))
    if len(two_or_more) == 0:
        print("No 2-agreement bit flips found. Nothing to do.")
        return

    lst_sorted_final = torch.tensor(sorted(all_pos), device='cuda:0')

    bit_positions = lst_sorted_final % 31
    index_positions = (lst_sorted_final - bit_positions) // 31

    bits_grouped = pd.DataFrame({
        'index': index_positions.cpu(),
        'bit': bit_positions.cpu()
    }).groupby('index', sort=False)['bit'].apply(list).to_dict()

    unique_indices = torch.tensor(list(bits_grouped.keys()), device='cuda:0')
    bit_flips = [torch.tensor(bits_grouped[idx.item()], device='cuda:0') for idx in unique_indices]

    mask = (t[unique_indices] != 0)
    keep_idx = mask.nonzero(as_tuple=False).flatten()

    if len(keep_idx) == 0:
        print("No non-zero values to flip. Exiting.")
        return

    unique_indices = unique_indices[keep_idx]
    bit_flips = [bit_flips[i] for i in keep_idx.tolist()]

    flipped_values = bitFLIP_v3_tensor(t[unique_indices], bit_flips)

    start = 0
    for name, param in model.named_parameters():
        if "weight" in name and "norm" not in name:
            param_size = param.numel()
            end = start + param_size
            mask = (unique_indices >= start) & (unique_indices < end)

            if mask.any():
                update_indices = unique_indices[mask] - start
                param_flat = param.view(-1).clone()

                non_zero_mask = param_flat[update_indices] != 0
                if non_zero_mask.any():
                    valid_update_indices = update_indices[non_zero_mask]
                    valid_flipped_values = flipped_values[mask][non_zero_mask]
                    param_flat[valid_update_indices] = valid_flipped_values

                param.data.copy_(param_flat.view(param.shape))

            start = end
#  Call the function
# fault_tolerance_two_agree(BER, model)
"""
Logger
"""
import csv
import os
import numpy as np  # For averaging multi-value metrics

csv_path = "results.csv"

# Initialize CSV (only once)
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["BER_power", "Iteration", "Accuracy", "Tacc", "Precision", "Recall", "Confidence", "Sub-Confidence", "Acc_50"])

# During each run:
def log_to_csv(power, iteration, accuracy, tacc, precision, recall, conf, sub_conf, acc_50):
    # Convert to scalar if necessary (e.g., take mean if tensor or list)
    def scalar(x):
        return x.mean().item() if hasattr(x, 'mean') else (np.mean(x) if isinstance(x, (list, tuple, np.ndarray)) else x)

    with open(csv_path, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            power,
            iteration,
            scalar(accuracy),
            scalar(tacc),
            scalar(precision),
            scalar(recall),
            scalar(conf),
            scalar(sub_conf),
            scalar(acc_50),
        ])
        f.flush()
        os.fsync(f.fileno())

# Code Block 9 Final boss
#doual
Accuracy=[]
Precision=[]
Recall=[]
Tacc=[]
conf=[]
sub_conf=[]
fault_position_array=[]
bits_array=[]
acc_50=[]
M=6
power=-6
while (power<-5):
  for i in range (2):
    print(power)
    BER=5*(10**power)
    #fault_position,bits=fault_positions(model,BER)
    fault_tolerance_two_agree( BER, model)
    return_acc,return_pre,return_rec,return_tacc,return_conf,return_sub_conf,return_acc_50=test(model)
    Accuracy.append(return_acc)
    Precision.append(return_pre)
    Recall.append(return_rec)
    Tacc.append(return_tacc)
    conf.append(return_conf)
    sub_conf.append(return_sub_conf)
    acc_50.append(return_acc_50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
    model = torch.load(path)
    model.eval()
    print(Accuracy)
    log_to_csv(power, i, return_acc, return_tacc, return_pre, return_rec, return_conf, return_sub_conf, return_acc_50)
  power+=1
print('all blocks converted Successfuly')