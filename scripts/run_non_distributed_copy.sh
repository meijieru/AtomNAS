#!/usr/bin/env bash

source ./scripts/utils/setup.sh

mkdir ~/data/
cp -r ${DATA_ROOT}/imagenet_lmdb ~/data/
export DATA_LMDB=~/data/imagenet_lmdb

python3 train.py app:$@
