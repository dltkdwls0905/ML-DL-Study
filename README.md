# 🧬 NGS Analysis Practice  
**NGS 변이 분석 파이프라인**을 한 번에 재현하는 Bash 기반 실습 도구입니다.  
페어드 FASTQ → **BWA-MEM 정렬** → **BAM 처리** → **변이 호출(VCF)** 을 자동화된 스크립트로 단계별 실행!

**BWA/SAMtools/BCFtools** 표준 체인 + IGV 시각화(옵션)

---

## 📌 주요 기능
✅ 예시 데이터(초소형 FASTQ) 자동 생성 → 빠르게 실습 가능  
✅ 참조 서열(FASTA) 준비 & 인덱싱 자동화  
✅ BWA-MEM 정렬 및 SAM→BAM 변환/정렬/중복제거/인덱싱  
✅ mpileup + bcftools call로 VCF 생성  
✅ IGV로 BAM/VCF 시각화(옵션)

---

## 🗂 프로젝트 구조
```
ngs-analysis-practice/
├─ ngs-analysis/
│  ├─ 00_setup_tools.sh      # 툴 설치/버전 확인
│  ├─ 01_get_data.sh         # 예시 FASTQ 생성(더미)
│  ├─ 02_index_ref.sh        # 레퍼런스 생성 + 인덱스
│  ├─ 03_align.sh            # BWA-MEM 정렬(FASTQ→SAM)
│  ├─ 04_bam_process.sh      # SAM→BAM, sort, markdup, index, QC
│  └─ 05_variant_call.sh     # mpileup + call → 최종 VCF
├─ docs/
│  └─ NGS_Pipeline.md        # 워크플로우 설명/메모
└─ linux-commands/
   └─ 01_basic_commands.sh   # 리눅스 기초 예시
```

---

## ⚙️ 설치 방법
### 1) 의존성(우분투/WSL 권장)
```bash
chmod +x ngs-analysis/*.sh
./ngs-analysis/00_setup_tools.sh   # bwa, samtools, bcftools 설치 + 버전 확인
```

### 2) 예시 데이터 & 레퍼런스 준비
```bash
./ngs-analysis/01_get_data.sh      # 초소형 페어드 FASTQ 생성
./ngs-analysis/02_index_ref.sh     # 참조 FASTA + 인덱스 생성
```

---

## 🚀 실행 방법
```bash
./ngs-analysis/03_align.sh         # 정렬: FASTQ → SAM
./ngs-analysis/04_bam_process.sh   # SAM→BAM, sort, markdup, index, flagstat
./ngs-analysis/05_variant_call.sh  # mpileup + call → VCF
```

**산출물(기본 경로: `data/out/`)**  
- `TS.sam`, `TS-sorted.bam`, `TS-dedup.bam`, `TS-dedup.bam.bai`  
- `TS-raw.bcf`, `TS-raw.vcf`, `TS-final.vcf`  
- `TS.flagstat.txt` (간단 QC)

---

## 👀 시각화(옵션, IGV)
1) IGV 실행 → **Genome**: `data/ref/SBA2018_Mit_Chl.fa` 로드  
2) **BAM**: `data/out/TS-dedup.bam` (+ `.bai`)  
3) **VCF**: `data/out/TS-final.vcf`  
4) 페어드 보기: 트랙 우클릭 → *View as Pairs*

---

## 🧰 기술 스택
- **Linux/WSL**, **Bash**  
- **bwa**, **samtools**, **bcftools** (표준 NGS 툴체인)  
- (옵션) **IGV** 시각화

---

## ⚠️ 주의사항
- Windows는 **WSL(우분투)** 권장. Git Bash도 가능하지만 패키지 설치 제약이 있을 수 있어요.  
- `samtools rmdup`는 최신판에서 **deprecated** → 본 프로젝트는 `bcftools markdup -r` 사용.  
- 실제 SRA 데이터를 사용할 땐 용량이 크니 `head`로 샘플링 후 대체하세요.

---

## 🔄 실제 데이터로 확장
```bash
# 예: SRA에서 받은 FASTQ를 data/raw에 배치하고 이름 맞추기
mv SRRXXXX_1.fastq.gz data/raw/TS-resized_1.fastq.gz
mv SRRXXXX_2.fastq.gz data/raw/TS-resized_2.fastq.gz

# 너무 크면 샘플링
zcat data/raw/TS-resized_1.fastq.gz | head -100000 | gzip > data/raw/tmp1.gz && mv data/raw/tmp1.gz data/raw/TS-resized_1.fastq.gz
zcat data/raw/TS-resized_2.fastq.gz | head -100000 | gzip > data/raw/tmp2.gz && mv data/raw/tmp2.gz data/raw/TS-resized_2.fastq.gz
```

---

## 📚 배운 점(요약)
- NGS 전처리의 **표준 흐름(FASTQ→BAM→VCF)**  
- **BWA/Samtools/Bcftools** 실전 사용  
- 재현 가능한 분석을 위한 **스크립트화/문서화** 습관

---

## ✅ 다음 단계(선택)
- `TS-final.vcf` → **Python(pandas/scikit-learn)으로 PCA/간단 분류** 예제 노트북 추가  
- 품질지표 강화: `samtools stats`, `qualimap` 등 연동  
- caller 비교: **FreeBayes**, **GATK HaplotypeCaller** 미니 파이프라인
