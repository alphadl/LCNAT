import codecs
import sys

from collections import defaultdict, OrderedDict
import h5py
from tqdm import tqdm

src_path = "e2d.trn.src.vcb"
tgt_path = "e2d.trn.trg.vcb"
align_path = "e2d.t3.final"

lcnat_src_path = "lc.en-de.en.vocab"
lcnat_tgt_path = "lc.en-de.de.vocab"

dict_src = defaultdict(str)
dict_tgt = defaultdict(str)
#dict_align = defaultdict(defaultdict(float))
dict_align = defaultdict(lambda: defaultdict(float))
# read the alignment vocab dict
with codecs.open(src_path, "r", "utf-8") as s:
    with codecs.open(tgt_path, "r", "utf-8") as t:
        s_list = [e.strip().split() for e in s.readlines()]
        t_list = [e.strip().split() for e in t.readlines()]
        for m in s_list:
            dict_src[m[0]] = m[1]
        for n in t_list:
            dict_tgt[n[0]] = n[1]
print("## Align: The length of src- and tgt- vocab are",len(dict_src), len(dict_tgt))
# read the alignments
with codecs.open(align_path, "r", "utf-8") as score:
    lines = score.readlines()
    for line in lines:        
        s_id = line.strip().split()[0]
        t_id = line.strip().split()[1]
        score = line.strip().split()[2]
        s_token = dict_src[s_id]
        t_token = dict_tgt[t_id]
        tmp_dict = defaultdict(float)
        tmp_dict[t_token] = float(score)
        dict_align[s_token][t_token] = float(score)
# 👆 loaded existing probs

# 👇 padding other positions
lcnat_dict_src = []
lcnat_dict_tgt = []
lcnat_dict_align = defaultdict(lambda: defaultdict(float))

with codecs.open(lcnat_src_path, "r", "utf-8") as s:
    with codecs.open(lcnat_tgt_path, "r", "utf-8") as t:
        s_list = [e.strip() for e in s.readlines()]
        t_list = [e.strip() for e in t.readlines()]
        for m in s_list:
            lcnat_dict_src.append(m)
        for n in t_list:
            lcnat_dict_tgt.append(n)
assert len(lcnat_dict_src) == len(lcnat_dict_tgt)
print("## MT: The length of src- and tgt- vocab are", len(lcnat_dict_src), len(lcnat_dict_tgt))

from tqdm import tqdm

# construct the final topology dict
for value_s in lcnat_dict_src:
    for value_t in lcnat_dict_tgt:
        # print(value_s, value_t)
        lcnat_dict_align[value_s][value_t] = dict_align[value_s][value_t]

# debug logging
tmp_sample_mt="tmp_mt.out"
tmp_sample_align="tmp_align.out"

with codecs.open(tmp_sample_mt, 'w+', 'utf-8') as tmp_mt:
    for ele in lcnat_dict_align[","]:
        tmp_mt.write(str(ele)+"\t"+str(lcnat_dict_align[","][ele])+"\n")

with codecs.open(tmp_sample_align, 'w+', 'utf-8') as tmp_align:
    for ele in dict_align[","]:
        tmp_align.write(str(ele)+"\t"+str(dict_align[","][ele])+"\n")

# soft_max
import numpy as np
def softmax(x, t=2):
    """Compute softmax values for each sets of scores in x."""
    # e_x = np.exp(np.array(x)/t - np.max(x))
    e_x = np.exp(np.array(x)/ t)
    return e_x / e_x.sum()

label_weights = np.zeros((len(lcnat_dict_align), len(lcnat_dict_align)), dtype=np.float32)

idx=0
for m in tqdm(lcnat_dict_align):
    weight = np.array(list(lcnat_dict_align[m].values()))
    label_weights[idx] = softmax(weight)
    idx += 1

f = h5py.File("clnat.en-de.h5", 'w')
f.create_dataset('weights', data=label_weights)
f.close()
def ave(x):
    """Compute average value for current list."""
    return sum(x)/len(x)