#!/usr/bin/env bash
set -euo pipefail

echo "[*] Update & basic build tools (Ubuntu/WSL 기준)"
if command -v apt >/dev/null 2>&1; then
  sudo apt update -y
  sudo apt upgrade -y
  sudo apt install -y build-essential git wget curl gzip bzip2 xz-utils       zlib1g zlib1g-dev libncurses5-dev libncursesw5-dev libbz2-dev liblzma-dev       default-jre default-jdk
fi

# 설치 방법 1) 배포판 패키지
if command -v apt >/dev/null 2>&1; then
  sudo apt install -y bwa samtools bcftools
fi

# 설치 확인
echo "[*] Versions"
command -v bwa && bwa 2>&1 | head -n 1 || true
command -v samtools && samtools --version | head -n 2 || true
command -v bcftools && bcftools --version | head -n 2 || true

echo "[*] Done."
