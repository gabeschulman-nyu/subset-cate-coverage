#!/bin/bash
# Fetch and pack the real base tables of the CFM-scale prior (experiments/cfm/prior.py).
#
#   benchmarks/get_openml_tables.sh [OUT]      (default: data/cfm; run from the repo root,
#                                               in the cfmdata environment)
#
# The tables are the OpenML datasets in CausalPFN's catalogue,
# assets/openml_catalogue.csv of github.com/vdblm/CausalPFN at commit 7da4afa.
# They are not shipped: each keeps its own OpenML licence. 337 datasets are
# listed; 7 have no Parquet file on OpenML and are skipped, none matches the
# filter that drops benchmark covariate sources (IHDP, ACIC, LaLonde, Twins,
# Jobs), and the remaining 330 are packed into OUT/openml_pack.{f32,json}, the
# path the training configs in configs/cfm/ name. About 70 MB of downloads.
#
# The pack is compared with the one the shipped heads were trained on. OpenML
# serves fixed dataset versions, so a mismatch means a file changed or failed
# to download; the script reports it and exits non-zero.
set -euo pipefail
OUT="${1:-data/cfm}"
PY="${PY:-python}"
COMMIT=7da4afa35affea0f1fb58893e9f41e87dafdc7eb
CATALOGUE_SHA=d6ccfa8dcc50983d308eaa0af642b154f93f33ef7bcd8555d7954fc82a98ff72
PACK_F32_SHA=d3f4bfce0e2e91520da9c1fafe880780491005fe14ceba15d4692ac0ad470662
PACK_JSON_SHA=720be950ec70886289c6ac587789b74f2ae522315ebd4775d6f696ea539f0da8

mkdir -p "$OUT"
curl -fsSL "https://raw.githubusercontent.com/vdblm/CausalPFN/$COMMIT/assets/openml_catalogue.csv" \
    -o "$OUT/openml_catalogue.csv"
echo "$CATALOGUE_SHA  $OUT/openml_catalogue.csv" | sha256sum -c -

"$PY" experiments/cfm/openml_tables.py "$OUT/openml_catalogue.csv" "$OUT/openml"
"$PY" -c "import sys; sys.path.insert(0, 'experiments/cfm'); from prior import pack_tables; \
print(pack_tables('$OUT/openml', '$OUT/openml_pack'))"

if printf '%s  %s\n%s  %s\n' "$PACK_F32_SHA" "$OUT/openml_pack.f32" "$PACK_JSON_SHA" "$OUT/openml_pack.json" \
    | sha256sum -c -; then
  echo "pack identical to the one the shipped heads were trained on"
else
  echo "the pack differs from the one the shipped heads were trained on" >&2
  exit 1
fi
