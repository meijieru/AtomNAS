#!/usr/bin/env bash

export DATA=~/data/imagenet
export DATA_LMDB=~/data/imagenet_lmdb

if [ ! -d "${DATA}" ]; then
    echo 'use remote imagenet'
    export DATA=${DATA_ROOT}/imagenet
fi
if [ ! -d "${DATA_LMDB}" ]; then
    echo 'use remote imagenet_lmdb'
    export DATA_LMDB=${DATA_ROOT}/imagenet_lmdb
fi

