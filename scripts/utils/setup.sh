#!/usr/bin/env bash

source ./scripts/utils/utils.sh

if [[ -z "${REMOTE_ID}" ]]; then
    green 'Not on REMOTE'
    export METIS_TASK_INDEX=0
    export METIS_WORKER_0_HOST=localhost
    export METIS_WORKER_0_PORT=9000
    export REMOTE_WORKER_NUM=1
    export REMOTE_OUTPUT=exp
	if [[ -z "${REMOTE_WORKER_GPU}" ]]; then
		red 'Please set ENV VAR ${REMOTE_WORKER_GPU}'
		exit 1
	fi
else
    echo 'On REMOTE'
    pip3 install torch torchvision
    source ./scripts/utils/setup_network.sh
fi

green "METIS_CONFIG: $METIS_WORKER_0_HOST:$METIS_WORKER_0_PORT, $METIS_TASK_INDEX"
green "NUM_WORKER: $REMOTE_WORKER_NUM, WORKER_GPU: $REMOTE_WORKER_GPU"
green "OUTPUT_DIR: ${REMOTE_OUTPUT}"
