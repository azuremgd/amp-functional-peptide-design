#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python predict.py --mode rescore --out results.csv "$@"
