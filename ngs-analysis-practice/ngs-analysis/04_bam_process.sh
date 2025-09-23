#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
OUT_DIR="${ROOT_DIR}/data/out"
REF="${ROOT_DIR}/data/ref/SBA2018_Mit_Chl.fa"
mkdir -p "${OUT_DIR}"

SAM="${OUT_DIR}/TS.sam"
BAM="${OUT_DIR}/TS.bam"
SORTED_BAM="${OUT_DIR}/TS-sorted.bam"
DEDUP_BAM="${OUT_DIR}/TS-dedup.bam"

echo "[*] SAM → BAM"
samtools view -Sb "${SAM}" > "${BAM}"

echo "[*] 정렬"
samtools sort -o "${SORTED_BAM}" "${BAM}"

# 교육 자료에는 rmdup 사용 예가 있으나, 최신 samtools에서는 deprecated일 수 있습니다.
# 대안: bcftools markdup 또는 Picard MarkDuplicates 사용.
# 여기서는 bcftools markdup로 예시
echo "[*] 중복 제거 (bcftools markdup)"
# 필요시 중간 파일 생성
samtools fixmate -m "${SORTED_BAM}" "${OUT_DIR}/TS-fixmate.bam"
samtools sort -o "${OUT_DIR}/TS-fixmate-sorted.bam" "${OUT_DIR}/TS-fixmate.bam"
bcftools markdup -r "${OUT_DIR}/TS-fixmate-sorted.bam" "${DEDUP_BAM}"

echo "[*] 인덱스 생성"
samtools index "${DEDUP_BAM}"

echo "[*] flagstat"
samtools flagstat "${DEDUP_BAM}" > "${OUT_DIR}/TS.flagstat.txt"
head -n 10 "${OUT_DIR}/TS.flagstat.txt" || true
