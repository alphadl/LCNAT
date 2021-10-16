#! /usr/bin/bash
ps aux|grep /root/miniconda2/envs/py3.7/bin/python|awk '{print $2}'|xargs kill -9

pip install -e ./fairseq/

set -e

s=$SRC
t=$TGT

if [ ! -d $checkpoint ]; then
  mkdir -p $checkpoint
fi

echo ">>> training"
python ./fairseq/fairseq/fairseq_cli/train.py ${databin} \
    --lcnat-weight-path clnat.en-de.h5 \
    --lcnat-src-vocab-path $databin/lc.en-de.en.vocab \
    --lcnat-tgt-vocab-path $databin/lc.en-de.de.vocab \
    --save-dir $checkpoint \
    -s $s -t $t \
    --ddp-backend=no_c10d --fp16 \
    --task levenshtein_transformer \
    --criterion lcnat_loss \
    --arch cmlm_transformer \
    --label-smoothing 0.1 \
    --attention-dropout 0.0 \
    --activation-dropout 0.0 \
    --dropout 0.2 \
    --noise random_delete \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 1e-07 --max-lr 1e-3 --lr-scheduler cosine \
    --warmup-init-lr 1e-07 --warmup-updates 10000 --lr-shrink 1 --lr-period-updates 290000 \
    --max-update 300000 \
    --weight-decay 0.0 --clip-norm 0.1 \
    --max-tokens 8192 --update-freq 4 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --no-progress-bar --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --seed 1 \
    --save-interval-updates 2000 \
    --keep-last-epochs 20 \
    --fp16-scale-tolerance 0.1