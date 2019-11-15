#!/usr/bin/env bash

source ./scripts/utils/setup.sh
source ./scripts/utils/export_data.sh
bash ./scripts/utils/remote_run.sh $@
