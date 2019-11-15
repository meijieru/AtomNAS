#!/usr/bin/env bash

if [[ -z "${REMOTE_ID}" ]]; then
    echo 'Not on REMOTE'
    export METIS_TASK_INDEX=0
    export METIS_WORKER_0_HOST=localhost
    export METIS_WORKER_0_PORT=9000
    export REMOTE_WORKER_NUM=1
    export REMOTE_OUTPUT=exp
else
    echo 'On REMOTE'
    pip3 install torch torchvision
    source ./scripts/utils/setup_network.sh
fi

echo METIS_CONFIG: $METIS_WORKER_0_HOST:$METIS_WORKER_0_PORT, $METIS_TASK_INDEX
echo NUM_WORKER: $REMOTE_WORKER_NUM, WORKER_GPU: $REMOTE_WORKER_GPU
echo OUTPUT_DIR: ${REMOTE_OUTPUT}
