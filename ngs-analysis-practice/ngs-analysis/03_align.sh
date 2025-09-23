#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
REF="${ROOT_DIR}/data/ref/SBA2018_Mit_Chl.fa"
RAW_DIR="${ROOT_DIR}/data/raw"
OUT_DIR="${ROOT_DIR}/data/out"
mkdir -p "${OUT_DIR}"

FASTQ1="${RAW_DIR}/TS-resized_1.fastq.gz"
FASTQ2="${RAW_DIR}/TS-resized_2.fastq.gz"

echo "[*] 정렬: bwa mem"
bwa mem "${REF}" "${FASTQ1}" "${FASTQ2}" > "${OUT_DIR}/TS.sam"

echo "[*] SAM top:"
head -n 20 "${OUT_DIR}/TS.sam" || true
