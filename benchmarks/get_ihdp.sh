#!/bin/bash
# Download the IHDP splits (ihdp_npci_1-100, as distributed with the CFR code
# of Shalit et al. 2017) into benchmarks/data/IHDP/ and check them against the
# files used for the paper.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/IHDP
for split in train test; do
  f=ihdp_npci_1-100.$split.npz
  [ -f "data/IHDP/$f" ] || curl -fL -o "data/IHDP/$f" "https://www.fredjo.com/files/$f"
done
cd data/IHDP
sha256sum -c <<'SUMS'
750697c71b4f8d7a3aafff771b56a4ac4cd83ec649bf69afb04f8a5aee41a240  ihdp_npci_1-100.train.npz
a70a8acbcc4e8deb677cc9bf9e9dabeb17caaa37cdbb1d7ba06be7ffb929c41c  ihdp_npci_1-100.test.npz
SUMS
