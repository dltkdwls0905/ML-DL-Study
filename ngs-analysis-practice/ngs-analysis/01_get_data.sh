#!/usr/bin/env bash
set -euo pipefail

# 워킹 디렉토리 구성
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)
RAW_DIR="${ROOT_DIR}/data/raw"
REF_DIR="${ROOT_DIR}/data/ref"
OUT_DIR="${ROOT_DIR}/data/out"
mkdir -p "${RAW_DIR}" "${REF_DIR}" "${OUT_DIR}"

echo "[*] 샘플 FASTQ(페어드) 더미 생성 (실습용 소형)"
# 실제 강의에선 head로 줄인 파일을 사용하지만, 여기서는 더미 예시 생성
cat > "${RAW_DIR}/TS-resized_1.fastq" <<'EOF'
@SEQ_ID/1
ACGTACGTACGTACGTACGT
+
FFFFFFFFFFFFFFFFFFFF
@SEQ_ID/1
ACGTACGTACGTACGTACGA
+
FFFFFFFFFFFFFFFFFFFF
EOF

cat > "${RAW_DIR}/TS-resized_2.fastq" <<'EOF'
@SEQ_ID/2
TCGTACGTACGTACGTACGT
+
FFFFFFFFFFFFFFFFFFFF
@SEQ_ID/2
TCGTACGTACGTACGTACGA
+
FFFFFFFFFFFFFFFFFFFF
EOF

gzip -f "${RAW_DIR}/TS-resized_1.fastq"
gzip -f "${RAW_DIR}/TS-resized_2.fastq"

echo "[*] 완료: ${RAW_DIR} 에 FASTQ 생성"
ls -lh "${RAW_DIR}"
