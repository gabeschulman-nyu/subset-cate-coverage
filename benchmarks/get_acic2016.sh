#!/bin/bash
# Generate the ACIC 2016 benchmark data into benchmarks/data/ACIC2016/.
# Needs R (>= 4.0) and git. Installs the official generator, aciccomp2016
# (github.com/vdorie/aciccomp), into benchmarks/.Rlib if it is not installed.
set -euo pipefail
cd "$(dirname "$0")"
export R_LIBS="$PWD/.Rlib${R_LIBS:+:$R_LIBS}"
mkdir -p .Rlib
if ! Rscript -e 'quit(status = !requireNamespace("aciccomp2016", quietly = TRUE))'; then
  tmp=$(mktemp -d)
  git clone -q --depth 1 https://github.com/vdorie/aciccomp.git "$tmp/aciccomp"
  R CMD INSTALL -l .Rlib "$tmp/aciccomp/2016"
  rm -rf "$tmp"
fi
Rscript get_acic2016.R
