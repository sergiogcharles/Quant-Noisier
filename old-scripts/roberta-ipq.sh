TOTAL_NUM_UPDATES=6108  # 2036 updates for each iteration
WARMUP_UPDATES=122
LR=2e-05
NUM_CLASSES=2
MAX_SENTENCES=16
ROBERTA_PATH=checkpoint/roberta/quant-noise-rte
SAVE_DIR=checkpoint/roberta/quant-noise-ipq
MNLI_PATH=multinli
CONFIG_PATH=fairseq/examples/quant_noise/transformer_quantization_config.yaml
fairseq-train --task sentence_prediction $MNLI_PATH \
    --restore-file $ROBERTA_PATH \
    --save-dir $SAVE_DIR \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 --lr-scheduler polynomial_decay \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --no-progress-bar --skip-invalid-size-inputs-valid-test --ddp-backend legacy_ddp \
    --quantization-config-path $CONFIG_PATH
