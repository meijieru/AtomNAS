#!/usr/bin/env bash

source ./scripts/utils/setup.sh
source ./scripts/utils/export_data.sh
python3 train.py app:$@
