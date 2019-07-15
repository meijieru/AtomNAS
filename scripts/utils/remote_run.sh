#!/usr/bin/env bash

if [[ -z "${ATOMNAS_VAL}" ]]; then
    export MAIN_FILE=train.py
else
    export MAIN_FILE=val.py
fi

NCCL_DEBUG=INFO \
python3 -m torch.distributed.launch --nproc_per_node=${ARNOLD_WORKER_GPU} --nnodes=${ARNOLD_WORKER_NUM} \
    --node_rank=$METIS_TASK_INDEX --master_addr=$METIS_WORKER_0_HOST --master_port=$METIS_WORKER_0_PORT \
    --use_env ${MAIN_FILE} app:$@
