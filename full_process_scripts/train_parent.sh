warmup=10000
seed=1
dropout=0.1
max_epoch=200
lr=0.001
act_drop=0.1
attn_drop=0.1
s=$1
t=$2
teacher_data=$3
keep_dir=revise/$s-$t-seed-$seed-lr-$lr-warmup-$warmup-dropout-$dropout-max_epoch-$max_epoch-act_drop-$act_drop-attn_drop-$attn_drop
logdir=tensorboard_save_non_early_stop/$keep_dir
save_dir=checkpoints/$keep_dir
CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU fairseq-train \
    $teacher_data \
    -s $s -t $t \
    --arch transformer --share-all-embeddings \
    --activation-dropout $act_drop --attention-dropout $attn_drop \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr $lr --lr-scheduler inverse_sqrt --warmup-updates $warmup --warmup-init-lr 1e-07 \
    --dropout $dropout --weight-decay 0.0 \
    --tensorboard-logdir $logdir \
    --save-dir $save_dir --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 \
    --update-freq 32 \
    --seed $seed \
    --max-epoch $max_epoch \
    --no-epoch-checkpoints \
    --keep-last-epochs 5 \
    --keep-best-checkpoints 5 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "lenpen":1.0}' \
    --eval-tokenized-bleu \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --fp16 2>&1