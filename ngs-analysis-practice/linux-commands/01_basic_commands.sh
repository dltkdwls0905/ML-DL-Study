#!/usr/bin/env bash
set -euo pipefail

pwd
ls -al
mkdir -p demo && cd demo
echo "Hello Bioinformatics" > hello.txt
cat hello.txt
