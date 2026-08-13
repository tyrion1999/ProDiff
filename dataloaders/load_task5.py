import re
from collections import OrderedDict, defaultdict

def load_list(p):
    with open(p, "r") as f:
        return [x.strip() for x in f if x.strip()]

def patient_from_slice(s):
    # e.g., prostate_29_slice_11 -> prostate_29
    m = re.match(r"(prostate_\d+)_slice_\d+", s)
    return m.group(1) if m else s.split("_slice_")[0]

def patient_order(lines):
    seen = set()
    order = []
    for l in lines:
        p = patient_from_slice(l)
        if p not in seen:
            seen.add(p)
            order.append(p)
    return order

def build_patient_slices(lines):
    d = defaultdict(list)
    for l in lines:
        d[patient_from_slice(l)].append(l)
    return d

# ====== 改成你自己仓库里 split1/split2 的真实路径 ======
split1_train = "/workspace/dataset/PROSTATE/Task05_split1(prostate)/train_slices.list"
split2_train = "/workspace/dataset/PROSTATE/Task05_split2(prostate)/train_slices.list"
split1_val   = "/workspace/dataset/PROSTATE/Task05_split1(prostate)/val.list"
split2_val   = "/workspace/dataset/PROSTATE/Task05_split2(prostate)/val.list"

s1_train = load_list(split1_train)
s2_train = load_list(split2_train)
s1_val = set(load_list(split1_val))
s2_val = set(load_list(split2_val))

# merged 方案：train = union(train patients), val = intersection(val patients)
d1 = build_patient_slices(s1_train)
d2 = build_patient_slices(s2_train)

order1 = patient_order(s1_train)
order2 = patient_order(s2_train)
merged_train_patients = order1 + [p for p in order2 if p not in set(order1)]  # 27 patients
merged_val_patients = sorted(list(s1_val & s2_val))  # 5 patients

# 生成 merged train_slices.list（按病人拼接；优先用 split1 里该病人的 slices，没有则用 split2）
merged_train_slices = []
for p in merged_train_patients:
    merged_train_slices += (d1[p] if p in d1 else d2[p])

# ====== 输出文件 ======
out_dir = "/workspace/dataset/PROSTATE/split_merged"
import os
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "train_slices.list"), "w") as f:
    f.write("\n".join(merged_train_slices) + "\n")

with open(os.path.join(out_dir, "val.list"), "w") as f:
    f.write("\n".join(merged_val_patients) + "\n")

with open(os.path.join(out_dir, "test.list"), "w") as f:
    f.write("\n".join(merged_val_patients) + "\n")

print("merged train patients:", len(merged_train_patients))
print("merged val patients:", len(merged_val_patients))
print("merged train slices:", len(merged_train_slices))
print("val patients:", merged_val_patients)
