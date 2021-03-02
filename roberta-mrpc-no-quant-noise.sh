TOTAL_NUM_UPDATES=2296
WARMUP_UPDATES=137
LR=1e-05
NUM_CLASSES=2
MAX_SENTENCES=4
ROBERTA_PATH=roberta_base/model.pt
MRPC_PATH=MRPC-bin/
SAVE_DIR=checkpoint/roberta/mrpc-no-quant-noise
UPDATE_FREQ=4

PYTHONPATH="~/Quant-Noisier/fairseq" python -m fairseq_cli.train $MRPC_PATH \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_base \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --ddp-backend legacy_ddp \
    --save-dir $SAVE_DIR \
    --update-freq $UPDATE_FREQ
