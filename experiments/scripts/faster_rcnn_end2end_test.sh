#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU DATASET NET [options args to {train,test}_net.py]
# DATASET is either pascal_voc / coco / imagenet
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end_test.sh 0 imagenet VGG_vid_test 100000\
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
ITERS=$4
NET_lc=${NET,,}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ;;
  imagenet)
    TRAIN_IMDB="imagenet_train"
    TEST_IMDB="imagenet_val"
    PT_DIR="imagenet"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG_DIR=experiments/logs
if [ ! -d "$LOG_DIR" ]; then
    mkdir "$LOG_DIR"
fi

LOG="$LOG_DIR/faster_rcnn_end2end_${NET}_${ITERS}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

WEIGHTS="output/faster_rcnn_end2end/${TRAIN_IMDB}/VGGnet_fast_rcnn_iter_${ITERS}.ckpt"

time python ./tools/test_net.py --gpu ${GPU_ID} \
  --weights ${WEIGHTS}\
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --network ${NET} \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x
