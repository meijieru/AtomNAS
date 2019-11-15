#!/usr/bin/env bash

NCCL_DEBUG=INFO \
python3 -m torch.distributed.launch --nproc_per_node=${REMOTE_WORKER_GPU} --nnodes=${REMOTE_WORKER_NUM} \
    --node_rank=$METIS_TASK_INDEX --master_addr=$METIS_WORKER_0_HOST --master_port=$METIS_WORKER_0_PORT \
    train.py app:$@
