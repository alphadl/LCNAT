#!/usr/bin/env bash
# Decode with Mask-Predict (iterative refinement, beam 5).
# Usage: SRC=en TGT=de DATA=./data/ende_data/databin CHECKPOINT=./checkpoint/ende/mask [SUBSET=valid] bash eval_mask.sh
set -e
SRC="${SRC:-en}"
TGT="${TGT:-de}"
DATA="${DATA:?Set DATA=path/to/databin}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=path/to/checkpoint}"
SUBSET="${SUBSET:-valid}"
CKPT="${CKPT:-checkpoint_best.pt}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$CHECKPOINT/gen}"
mkdir -p "$OUT_DIR"

python "$ROOT/fairseq/fairseq_cli/generate.py" "$DATA" \
  --gen-subset "$SUBSET" \
  --path "$CHECKPOINT/$CKPT" \
  -s $SRC -t $TGT \
  --task translation_lev \
  --iter-decode-max-iter 10 \
  --iter-decode-eos-penalty 0 \
  --iter-decode-with-beam 5 \
  --iter-decode-force-max-iter \
  --remove-bpe \
  --batch-size 64 \
  > "$OUT_DIR/${SUBSET}.out" 2> "$OUT_DIR/${SUBSET}.log"
grep ^H "$OUT_DIR/${SUBSET}.out" | cut -f3- > "$OUT_DIR/${SUBSET}.hyp"
echo "Hypotheses: $OUT_DIR/${SUBSET}.hyp"
