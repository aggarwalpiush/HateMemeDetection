SELF=$(dirname "$(realpath $0)")
PRETRAIN="$SELF/../pretrain_model"
CKPT="$SELF/../checkpoints"
VLBERT_CKPT="$CKPT/vl-bert"
DATA="$SELF/../data"

mkdir -p "$VLBERT_CKPT"
docker run --gpus all \
    -v "$SELF":/src \
    dsfhe49854/vl-bert \
    sh /src/scripts/init.sh

# MODEL_NAME="vl-bert-large-v4"
# CONFIG_NAME="large_4x14G_fp32_k8s_v4"
# CKPT_NAME="vl-bert_large_res101_cls-0009.model"

# MODEL_NAME="vl-bert-large-v5-emotion"
# CONFIG_NAME="large_4x14G_fp32_k8s_v5_emotion"
# CKPT_NAME="vl-bert_large_res101_cls-0009.model"

MODEL_NAME="vl-bert-large-v5-race"
CONFIG_NAME="large_4x14G_fp32_k8s_v5_race"
CKPT_NAME="vl-bert_large_res101_cls-0009.model"

# MODEL_NAME="vl-bert-large-v5-race-emotion"
# CONFIG_NAME="large_4x14G_fp32_k8s_v5_race_emotion"
# CKPT_NAME="vl-bert_large_res101_cls-0009.model"

echo "[$MODEL_NAME] & [$CONFIG_NAME] --> [$CKPT_NAME]"

if [ ! -d "$VLBERT_CKPT/$MODEL_NAME" ]; then
    mkdir -p "$VLBERT_CKPT/$MODEL_NAME"
    echo "************  [TRAIN] $MODEL_NAME  ************"

    docker run --gpus all \
        --shm-size 4G \
        -v $SELF:/src \
        -v $PRETRAIN:/pretrain_model \
        -v $VLBERT_CKPT:/checkpoints \
        -v $DATA:/data \
        dsfhe49854/vl-bert \
        sh /src/scripts/dist_run_single.sh 4 cls/train_end2end.py \
        "/src/cfgs/cls/$CONFIG_NAME.yaml" "/checkpoints/$MODEL_NAME"
else
    echo "Dir \"$VLBERT_CKPT/$MODEL_NAME\" is already exist, assume training is completed..."
fi;

echo "Training is completed"

if [ -e "$VLBERT_CKPT/$MODEL_NAME/$CONFIG_NAME/train1_train/$CKPT_NAME" ] && [ ! -e "$VLBERT_CKPT/$MODEL_NAME/${CONFIG_NAME}_cls_test.csv" ]; then
    echo "************  [TEST] $MODEL_NAME  ************"
    docker run --gpus all \
        --shm-size 4G \
        -v $SELF:/src \
        -v $PRETRAIN:/pretrain_model \
        -v $VLBERT_CKPT:/checkpoints \
        -v $DATA:/data \
        dsfhe49854/vl-bert \
        python3 /src/cls/test.py \
        --cfg "/src/cfgs/cls/$CONFIG_NAME.yaml" \
        --ckpt "/checkpoints/$MODEL_NAME/$CONFIG_NAME/train1_train/$CKPT_NAME" \
        --result-path "/checkpoints/$MODEL_NAME"
else
    echo "Checkpoint not found: $VLBERT_CKPT/$MODEL_NAME/$CONFIG_NAME/train1_train/$CKPT_NAME"
fi;

echo "Testing results are generated"