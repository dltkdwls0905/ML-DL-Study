# NGS Analysis Practice (Starter)

이 저장소는 FASTQ → SAM/BAM → VCF까지의 **NGS 변이분석 파이프라인**을 Bash 스크립트로 재현하기 위한 스타터입니다.  
로컬/WSL/리눅스에서 바로 실행할 수 있도록 단계별 스크립트로 분리했습니다.

## 구성
- `ngs-analysis/00_setup_tools.sh` : 필수 도구 설치(PATH 설정 포함)
- `ngs-analysis/01_get_data.sh`    : 예시 FASTQ(샘플용) 및 폴더 구성
- `ngs-analysis/02_index_ref.sh`   : 레퍼런스 FASTA 준비 및 인덱스
- `ngs-analysis/03_align.sh`       : BWA-MEM으로 정렬 (FASTQ → SAM)
- `ngs-analysis/04_bam_process.sh` : SAM→BAM, 정렬, 중복제거, 인덱싱
- `ngs-analysis/05_variant_call.sh`: samtools mpileup + bcftools로 VCF 생성
- `docs/NGS_Pipeline.md`           : 워크플로우 설명 및 실행 예시

## 빠른 시작
```bash
# 0) 실행 권한 부여
chmod +x ngs-analysis/*.sh

# 1) 필수 도구 설치 (필요 시 sudo 비밀번호 입력)
./ngs-analysis/00_setup_tools.sh

# 2) 예시 데이터/폴더 구성
./ngs-analysis/01_get_data.sh

# 3) 레퍼런스 준비 및 인덱싱
./ngs-analysis/02_index_ref.sh

# 4) 정렬
./ngs-analysis/03_align.sh

# 5) BAM 처리
./ngs-analysis/04_bam_process.sh

# 6) 변이 호출
./ngs-analysis/05_variant_call.sh
```

> 참고: Windows라면 WSL(우분투) 또는 Git Bash에서 실행하세요. IGV 시각화는 GUI 환경에서 진행합니다.
