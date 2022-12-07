warmup=1000
seed=1
lr=0.0002
bsz=1000
w_decay=0.0
max_epoch=200
attn_dropout=0.1
act_dropout=0.1
freq=1
student_data=$1
switch_checkpoint=$2
save_dir=checkpoints/freq-$freq-lr-$lr-bsz-$bsz-clip-base-clean-warmup-$warmup-max_epoch-$max_epoch-seed-$seed-attn-dropout-$attn_dropout-act-dropout-$act_dropout-w_decay-$w_decay
rm -r $save_dir
mkdir -p $save_dir
cp $switch_checkpoint $save_dir/checkpoint_last.pt
nvidia-smi
# --pretrained-mt-checkpoint $pretrained_mt_checkpoint \
#     --encoder-embed-path $encoder_embed_path \
CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU fairseq-train \
    $student_data \
    --arch transformer \
    --share-decoder-input-output-embed \
    --reset-optimizer \
    --reset-dataloader \
    --fp16 \
    --update-freq $freq \
    --max-tokens $bsz \
    --tensorboard-logdir clean_train \
    --save-dir $save_dir \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr $lr --lr-scheduler inverse_sqrt --warmup-updates $warmup --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay $w_decay \
    --attention-dropout $attn_dropout \
    --activation-dropout $act_dropout \
    --seed $seed \
    --eval-bleu \
    --max-epoch $max_epoch \
    --keep-last-epochs 1 \
    --keep-best-checkpoints 1 \
    --eval-bleu-args '{"beam": 5, "lenpen":1}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 2>&1


# fairseq-generate valid_17_test_18_8k_tc_all_joined-bin --path $save_dir/checkpoint_best.pt --remove-bpe --beam 5 --lenpen 1 --batch-size 50 --gen-subset test > test_out/test.share-all-lr-$lr-bsz-$bsz-warmup-$warmup-max_pos-$max_pos-patience-$patience-seed-$seed-greedy-eval-interval-$interval-attn-dropout-$attn_dropout-act-dropout-$act_dropout-left-pad-source.out
# fairseq-generate valid_17_test_18_8k_tc_all_joined-bin --path $save_dir/checkpoint_best.pt --remove-bpe --beam 5 --lenpen 1 --batch-size 50 --gen-subset valid > valid_out/valid.share-all-lr-$lr-bsz-$bsz-warmup-$warmup-max_pos-$max_pos-patience-$patience-seed-$seed-greedy-eval-interval-$interval-attn-dropout-$attn_dropout-act-dropout-$act_dropout-left-pad-source.out

# bash ../scripts/sacrebleu.sh wmt17 tr en valid_out/valid.share-all-lr-$lr-bsz-$bsz-warmup-$warmup-max_pos-$max_pos-patience-$patience-seed-$seed-greedy-eval-interval-$interval-attn-dropout-$attn_dropout-act-dropout-$act_dropout-left-pad-source.out
# bash ../scripts/sacrebleu.sh wmt18 tr en test_out/test.share-all-lr-$lr-bsz-$bsz-warmup-$warmup-max_pos-$max_pos-patience-$patience-seed-$seed-greedy-eval-interval-$interval-attn-dropout-$attn_dropout-act-dropout-$act_dropout-left-pad-source.out

# bash ../scripts/sacrebleu_detrucase.sh wmt17 tr en valid_out/valid.share-all-lr-$lr-bsz-$bsz-warmup-$warmup-max_pos-$max_pos-patience-$patience-seed-$seed-greedy-eval-interval-$interval-attn-dropout-$attn_dropout-act-dropout-$act_dropout-left-pad-source.out
# bash ../scripts/sacrebleu_detrucase.sh wmt18 tr en test_out/test.share-all-lr-$lr-bsz-$bsz-warmup-$warmup-max_pos-$max_pos-patience-$patience-seed-$seed-greedy-eval-interval-$interval-attn-dropout-$attn_dropout-act-dropout-$act_dropout-left-pad-source.out