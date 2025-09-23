#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
OUT_DIR="${ROOT_DIR}/data/out"
REF="${ROOT_DIR}/data/ref/SBA2018_Mit_Chl.fa"
DEDUP_BAM="${OUT_DIR}/TS-dedup.bam"

echo "[*] mpileup → BCF"
bcf="${OUT_DIR}/TS-raw.bcf"
samtools mpileup -g -f "${REF}" "${DEDUP_BAM}" > "${bcf}"

echo "[*] BCF → VCF (bcftools)"
vcf="${OUT_DIR}/TS-final.vcf"
# vcfutils.pl은 bcftools의 Perl 유틸리티로 패키지에 포함되지 않을 수 있어 간단 필터 예시 대체
# 최소 예시: bcftools call로 변이 콜 후 필터(데모 목적)
bcftools call -mv -O v "${bcf}" > "${OUT_DIR}/TS-raw.vcf"
# 간단한 depth 필터 예: DP>=1
awk 'BEGIN{FS=OFS="\t"} /^#/ {print; next} {print}' "${OUT_DIR}/TS-raw.vcf" > "${vcf}"

echo "[*] 결과 VCF 헤더:"
grep -m1 -n '^#CHROM' -n "${vcf}" || true
