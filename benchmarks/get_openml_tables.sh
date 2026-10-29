#!/bin/bash
# Fetch the OpenML base tables of the prior in experiments/cfm/prior.py and pack
# them into OUT/openml_pack.{f32,json}. Run from the repo root in the cfmdata
# environment; about 70 MB of downloads.
#
#   benchmarks/get_openml_tables.sh [OUT]      (default: data/cfm)
#
# The tables are those in CausalPFN's catalogue (assets/openml_catalogue.csv at
# commit 7da4afa). Of its 337 datasets, 7 have no Parquet file on OpenML and
# are skipped. The tables are under their OpenML licences and are not
# redistributed. The script exits non-zero if the pack differs from the one
# the heads were trained on.
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
  echo "pack identical to the one the heads were trained on"
else
  echo "the pack differs from the one the heads were trained on" >&2
  exit 1
fi
