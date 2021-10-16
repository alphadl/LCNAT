#! /usr/bin/bash

s=$1
t=$2
task=$3

if [ ! -d ./rst/${s}${t}/${task} ]; then
  mkdir -p ./rst/${s}${t}/${task}
fi

echo ">>> validating"

for file in ./checkpoint/${s}${t}/${task}/*.pt
do
  filename=$(basename $file)
  echo ${filename} 'Translating...'
  CUDA_VISIBLE_DEVICES=$4 python ./fairseq_mask/fairseq/fairseq_cli/generate.py $TASK_path \
  --gen-subset valid \
  --task translation_lev \
  --path ./checkpoint/${s}${t}/${task}/${filename} \
  --iter-decode-max-iter 10 \
  --iter-decode-eos-penalty 0 \
  --remove-bpe \
  --iter-decode-with-beam 5 \
  --print-step \
  --iter-decode-force-max-iter \
  --batch-size 30 > ./rst/${s}${t}/${task}/${filename}.out 2>&1 &
  wait
done