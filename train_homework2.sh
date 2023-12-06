#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


set -x
set -e

DESC=${@:1}

set -o pipefail

ARGS=$(cat $1)
OUTPUT_BASE=$(echo $1 | sed -e "s/configs/exps/g" | sed -e "s/.args$//g")
unset DISPLAY
mkdir -p $OUTPUT_BASE

for RUN in $(seq 1000); do
  ls $OUTPUT_BASE | grep -q run$RUN && continue
  OUTPUT_DIR=$OUTPUT_BASE/run$RUN
  mkdir $OUTPUT_DIR && break
done

# run backup
echo "Backing up to log dir: $OUTPUT_DIR"
cp -r $1 train_homework2.py $OUTPUT_DIR
cp -r rcnn_model.py $OUTPUT_DIR
cp -r hw2_data_loador.py $OUTPUT_DIR
echo " ...Done"


pushd $OUTPUT_DIR
echo $DESC > desc
echo " ...Done"

python3 train_homework2.py $ARGS |& tee output.log
