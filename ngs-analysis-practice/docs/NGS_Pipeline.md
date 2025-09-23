# NGS Pipeline Workflow (개요)

본 워크플로우는 공개 강의 실습 흐름을 바탕으로 **BWA → samtools → bcftools** 조합으로 변이(SNP) 탐지까지 재현합니다.

## 단계
1. **Raw data 준비**: 샘플 FASTQ(페어드) 준비 및 내용 확인
2. **Reference 준비**: 레퍼런스 FASTA와 인덱스 파일 준비
3. **정렬(Alignment)**: `bwa mem`으로 FASTQ→SAM
4. **BAM 처리**: `samtools`로 SAM→BAM 변환, 정렬, 중복제거, 인덱싱
5. **변이 호출**: `samtools mpileup` + `bcftools`로 VCF 생성
6. **시각화**: IGV로 BAM/VCF 확인(옵션)

각 단계는 `ngs-analysis/*.sh` 스크립트로 분리되어 있습니다.
