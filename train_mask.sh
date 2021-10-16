#!/usr/bin/env bash
# Train Mask-Predict (CMLM) with LCNAT data-dependent prior.
# Required env: SRC, TGT, databin (path to binarized data), checkpoint (save dir), lcnat_weight_path (.h5).
# Example: SRC=en TGT=de databin=./data/ende_data/databin checkpoint=./checkpoint/ende/mask lcnat_weight_path=./data/ende_data/prior.h5 bash train_mask.sh
set -e
SRC="${SRC:-en}"
TGT="${TGT:-de}"
DATABIN="${databin:?Set databin=path/to/databin}"
CHECKPOINT="${checkpoint:?Set checkpoint=path/to/save}"
LCNAT_WEIGHT="${lcnat_weight_path:-$DATABIN/../prior.h5}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$CHECKPOINT"
pip install -e "$ROOT/fairseq/" --quiet 2>/dev/null || true

python "$ROOT/fairseq/fairseq_cli/train.py" "$DATABIN" \
  --save-dir "$CHECKPOINT" \
  -s $SRC -t $TGT \
  --task translation_lev \
  --criterion lcnat_loss \
  --arch cmlm_transformer \
  --lcnat-weight-path "$LCNAT_WEIGHT" \
  --lcnat-src-vocab-path "$DATABIN/lc.$SRC-$TGT.$SRC.vocab" \
  --lcnat-tgt-vocab-path "$DATABIN/lc.$SRC-$TGT.$TGT.vocab" \
  --label-smoothing 0.1 \
  --attention-dropout 0.0 \
  --activation-dropout 0.0 \
  --dropout 0.2 \
  --noise random_mask \
  --share-decoder-input-output-embed \
  --decoder-learned-pos \
  --encoder-learned-pos \
  --apply-bert-init \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr 1e-7 --max-lr 1e-3 --lr-scheduler cosine \
  --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --lr-period-updates 290000 --max-update 300000 \
  --weight-decay 0.0 --clip-norm 0.1 \
  --max-tokens 8192 --update-freq 4 \
  --fp16 --ddp-backend no_c10d \
  --save-interval-updates 2000 \
  --keep-last-epochs 20 \
  --seed 1 \
  --log-format simple --log-interval 100
