#!/usr/bin/env bash
# Run test-set decoding and BLEU (Levenshtein). Set SRC, TGT, DATA, CHECKPOINT, REF.
set -e
SRC="${SRC:-en}"
TGT="${TGT:-de}"
DATA="${DATA:?Set DATA=path/to/databin}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=path/to/checkpoint}"
REF="${REF:-}"
SUBSET=test
export DATA CHECKPOINT SRC TGT SUBSET
bash "$(dirname "${BASH_SOURCE[0]}")/eval_lev.sh"
if [ -n "$REF" ] && [ -f "$REF" ]; then
  sacrebleu "$REF" -i "$CHECKPOINT/gen/test.hyp" -b
fi
