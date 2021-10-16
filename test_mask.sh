task=$1
python ./fairseq_mask/scripts/average_checkpoints.py --inputs ./checkpoint/${s}${t}/${task}/${filename}/top5_checkpoints_path/* --output ./checkpoint/${s}${t}/${task}/${filename}/top5.pt

CUDA_VISIBLE_DEVICES=$2 python ./fairseq_lev/fairseq/fairseq_cli/generate.py $TASK_path \
  --gen-subset test \
  --task translation_lev \
  --path ./checkpoint/${s}${t}/${task}/${filename}/top5.pt \
  --iter-decode-max-iter 10 \
  --iter-decode-eos-penalty 0 \
  --remove-bpe \
  --iter-decode-with-beam 5 \
  --print-step \
  --iter-decode-force-max-iter \
  --batch-size 30 > ./rst/${s}${t}/${task}/top5.pt.out 2>&1 &

bash ./fairseq_mask/scripts/compound_split_bleu.sh ./rst/${s}${t}/${task}/top5.pt.out
# ~27.8