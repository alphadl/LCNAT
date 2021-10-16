#! /usr/bin/bash

# preprocess data
s=en
t=de

echo ">>> binarize the data"

if [ ! -d ./data/${s}${t}_data/databin/sentence ]; then
  mkdir -p ./data/${s}${t}_data/databin/sentence
  
  nohup python ./fairseq/fairseq_cli/preprocess.py \
    --source-lang ${s} --target-lang ${t} \
    --trainpref ./data/${s}${t}_data/train_kd \
    --validpref ./data/${s}${t}_data/valid --testpref ./data/${s}${t}_data/test \
    --joined-dictionary \
    --destdir ./data/${s}${t}_data/databin/ \
    --workers 64
fi
wait
echo ">>> binarizing finished"

python ./run/lcnat_extract_vocab.py ./data/${s}${t}_data/databin/dict.en.txt  ./data/${s}${t}_data/databin/lc.en-de.en.vocab
python ./run/lcnat_extract_vocab.py ./data/${s}${t}_data/databin/dict.de.txt  ./data/${s}${t}_data/databin/lc.en-de.de.vocab
