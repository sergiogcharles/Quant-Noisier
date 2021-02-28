TOTAL_UPDATES=3000
WARMUP_UPDATES=500
PEAK_LR=0.0005
TOKENS_PER_SAMPLE=512
MAX_POSITIONS=512
MAX_SENTENCES=2
UPDATE_FREQ=16
DATA_DIR=data-bin/wikitext-103
RESTORE_DIR=roberta_base/model.pt
SAVE_DIR=checkpoint/roberta/FAKE-TEST-CLI

PYTHONPATH="~/Quant-Noisier/fairseq" python -m fairseq_cli.train $DATA_DIR \
    --task masked_lm --criterion masked_lm --arch roberta_base \
    --sample-break-mode complete \
    --tokens-per-sample $TOKENS_PER_SAMPLE --max-positions $MAX_POSITIONS \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR \
    --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES \
    --update-freq $UPDATE_FREQ --max-update $TOTAL_UPDATES \
    --save-dir $SAVE_DIR \
    --ddp-backend legacy_ddp --encoder-layerdrop 0.2 \
    --quant-noise-scalar 0.5 --untie-weights-roberta \
    --restore-file $RESTORE_DIR \
    --bits 1
