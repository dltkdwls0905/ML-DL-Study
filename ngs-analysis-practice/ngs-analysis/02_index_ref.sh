#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
REF_DIR="${ROOT_DIR}/data/ref"
mkdir -p "${REF_DIR}"

echo "[*] 레퍼런스 FASTA 생성(데모 용)"
cat > "${REF_DIR}/SBA2018_Mit_Chl.fa" <<'EOF'
>chrM
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT
>chrC
TTTTACGTACGTACGTACGTACGTACGTACGTACGTAAAA
EOF

echo "[*] bwa 인덱스 생성"
bwa index "${REF_DIR}/SBA2018_Mit_Chl.fa"

echo "[*] samtools faidx 생성"
samtools faidx "${REF_DIR}/SBA2018_Mit_Chl.fa"

echo "[*] 완료: ${REF_DIR} 에 인덱스 파일 생성"
ls -lh "${REF_DIR}"
