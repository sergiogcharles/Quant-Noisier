#!/bin/bash                                                                                                                                                                                          

seeds=(2 3 1)
for seed in "${seeds[@]}"; do
    TOTAL_NUM_UPDATES=2296
    WARMUP_UPDATES=137
    LR=1e-05
    NUM_CLASSES=2
    MAX_SENTENCES=1
    ROBERTA_PATH=roberta_base/model.pt
    RTE_PATH=RTE-bin/
    UPDATE_FREQ=16
    SAVE_DIR="checkpoint/roberta/new-adaptive-mrpc-scalar-4-quant-noise-seed-$seed"
    echo "running with seed $seed"
    echo "saving to $SAVE_DIR"

    PYTHONPATH="~/Quant-Noisier/fairseq" python -m fairseq_cli.train $RTE_PATH \
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
        --quant-noise-scalar 0.5 \
        --save-dir $SAVE_DIR \
        --bits 4 \
        --update-freq $UPDATE_FREQ \
        --quant-noise-adaptive-new True \
        --loss-file "losses.csv" \
        --lamb 0.125 \
        --seed $seed
    
    rm $SAVE_DIR/checkpoint1.pt
    rm $SAVE_DIR/checkpoint2.pt
    rm $SAVE_DIR/checkpoint3.pt
    rm $SAVE_DIR/checkpoint4.pt
    rm $SAVE_DIR/checkpoint5.pt
    rm $SAVE_DIR/checkpoint6.pt
    rm $SAVE_DIR/checkpoint7.pt
    rm $SAVE_DIR/checkpoint8.pt
    rm $SAVE_DIR/checkpoint9.pt
    rm $SAVE_DIR/checkpoint10.pt
done
