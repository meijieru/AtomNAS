#!/usr/bin/env bash

source ./scripts/utils/utils.sh

if [[ -z "${ARNOLD_ID}" ]]; then
    green 'Not on Arnold'
    export METIS_TASK_INDEX=0
    export METIS_WORKER_0_HOST=localhost
    export METIS_WORKER_0_PORT=9000
    export ARNOLD_WORKER_NUM=1
    export ARNOLD_OUTPUT=exp
    if [[ -z "${ARNOLD_WORKER_GPU}" ]]; then
	red 'Please set ENV VAR ${ARNOLD_WORKER_GPU}'
	exit 1
    fi
    if [[ -z "${DATA_ROOT}" ]]; then
	red '${DATA_ROOT} Fallback to home directory'
	export DATA_ROOT=${HOME}
    fi
else
    echo 'On Arnold'
    pip3 install torch torchvision
    source ./scripts/utils/setup_network.sh
fi

green "METIS_CONFIG: $METIS_WORKER_0_HOST:$METIS_WORKER_0_PORT, $METIS_TASK_INDEX"
green "NUM_WORKER: $ARNOLD_WORKER_NUM, WORKER_GPU: $ARNOLD_WORKER_GPU"
green "OUTPUT_DIR: ${ARNOLD_OUTPUT}"
