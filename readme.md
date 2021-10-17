# LCNAT

Code for **"Understanding and Improving Lexical Choice in Non-Autoregressive Translation"** (ICLR 2021).

We add a data-dependent prior (from word alignment) to NAT training: an extra KL term with λ decay over the first half of training.

Paper: [OpenReview](https://openreview.net/pdf?id=ZTFeSBIX9C)

## Setup

```bash
git clone https://github.com/alphadl/LCNAT.git
cd LCNAT
pip install -e fairseq/
```

Dependencies: PyTorch, fairseq (included), h5py, numpy.

## Data

1. Put KD data in e.g. `data/ende_data/`: `train_kd.{src,tgt}`, `valid.{src,tgt}`, `test.{src,tgt}` (BPE).
2. Binarize: `SRC=en TGT=de DATA_DIR=./data/ende_data bash preprocess.sh`
3. Build prior: matrix `[V_src, V_tgt]` in `.h5` with key `"weights"` (e.g. [Bottleneck_LC/scripts/build_prior.py](https://github.com/alphadl/Bottleneck_LC)). Save as `data/ende_data/prior.h5`.

## Training

```bash
SRC=en TGT=de databin=./data/ende_data/databin checkpoint=./checkpoint/ende/mask lcnat_weight_path=./data/ende_data/prior.h5 bash train_mask.sh
```

Levenshtein: same env with `train_lev.sh`.

## Eval

```bash
SRC=en TGT=de DATA=./data/ende_data/databin CHECKPOINT=./checkpoint/ende/mask bash eval_mask.sh
REF=./data/ende_data/test.de bash test_mask.sh
```

## Lexical choice accuracy

Word-level translation accuracy stratified by **source-word frequency** (H/M/L), as in the paper and RLFW-NAT Table 8: for each source token with aligned reference word, check if the hypothesis has the correct ref word at the aligned position; bucket by frequency estimated on source-side training data.

**Requirements:** Tokenized `src`, `ref`, `hyp` (one sentence per line); alignments in fast_align format (`i-j` per link, one line per sentence).

1. Align source–reference and reference–hypothesis (e.g. [fast_align](https://github.com/clab/fast_align)):
   ```bash
   fast_align -i test.en-de.raw -d -v -o > test.en-de.fwd
   # format: "src_idx-ref_idx" per sentence
   ```
   Do the same for ref–hyp (ref as first file, hyp as second) to get `test.de-hyp.align`.

2. Source-side frequency for H/M/L buckets: use training corpus or fairseq dict:
   ```bash
   python run/lexical_choice_accuracy.py \
     --src test.en --ref test.de --hyp test.hyp.de \
     --align-src-ref test.en-de.align --align-ref-hyp test.de-hyp.align \
     --dict-src data/ende_data/databin/dict.en.txt
   ```
   Or `--train-src data/ende_data/train_kd.en` to count tokens in the training source file.

Output: accuracy overall and per bucket (H / M / L).

## Citation

```bibtex
@inproceedings{ding2021understanding,
  title={Understanding and Improving Lexical Choice in Non-Autoregressive Translation},
  author={Ding, Liang and Wang, Longyue and Liu, Xuebo and Wong, Derek F. and Tao, Dacheng and Tu, Zhaopeng},
  booktitle={ICLR},
  year={2021}
}
```
