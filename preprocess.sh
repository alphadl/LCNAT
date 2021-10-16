#!/usr/bin/env bash
# Binarize KD data and extract vocab for LCNAT prior (WAD).
# Expects: data/ende_data/train_kd.{en,de}, valid.{en,de}, test.{en,de}
# Set SRC, TGT, DATA_DIR to override (e.g. SRC=en TGT=de DATA_DIR=./data/ende_data).
set -e
SRC="${SRC:-en}"
TGT="${TGT:-de}"
DATA_DIR="${DATA_DIR:-./data/${SRC}${TGT}_data}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKERS="${WORKERS:-8}"

mkdir -p "$DATA_DIR/databin"

if [ ! -f "$DATA_DIR/databin/dict.$SRC.txt" ]; then
  echo ">>> Binarizing..."
  python "$ROOT/fairseq/fairseq_cli/preprocess.py" \
    --source-lang $SRC --target-lang $TGT \
    --trainpref "$DATA_DIR/train_kd" \
    --validpref "$DATA_DIR/valid" \
    --testpref "$DATA_DIR/test" \
    --destdir "$DATA_DIR/databin" \
    --joined-dictionary \
    --workers $WORKERS
  echo ">>> Binarizing finished."
fi

python "$ROOT/run/lcnat_extract_vocab.py" \
  "$DATA_DIR/databin/dict.$SRC.txt" \
  "$DATA_DIR/databin/lc.$SRC-$TGT.$SRC.vocab"
python "$ROOT/run/lcnat_extract_vocab.py" \
  "$DATA_DIR/databin/dict.$TGT.txt" \
  "$DATA_DIR/databin/lc.$SRC-$TGT.$TGT.vocab"
echo ">>> Vocab saved to $DATA_DIR/databin/lc.*.vocab"
