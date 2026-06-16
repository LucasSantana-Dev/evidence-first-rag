#!/usr/bin/env bash
# eval/check.sh — run the eval suite and compare against baseline.json.
# Exit 0 if delta within tolerance, 1 if any metric regresses by >5pp.
# Designed to be called manually or chained from build.py after full reindex.
set -uo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL="${1:-rolling-$(date +%Y%m%d-%H%M)}"
TOL_PP="${TOL_PP:-5}"

cd "$EVAL_DIR"
DATASET="${RAG_EVAL_DATASET:-$EVAL_DIR/golden.jsonl}"
"$EVAL_DIR/../venv/bin/python3" run.py --dataset "$DATASET" --label "$LABEL" >/tmp/eval-run.out 2>&1
status=$?
cat /tmp/eval-run.out
[ "$status" -ne 0 ] && exit "$status"

CURRENT="$EVAL_DIR/${LABEL}.json"
BASELINE="${RAG_EVAL_BASELINE:-$EVAL_DIR/baseline-golden.json}"
[ -f "$BASELINE" ] || { echo "(no baseline.json — skipping comparison)"; exit 0; }
[ -f "$CURRENT" ]  || { echo "(no $CURRENT — eval may have failed)"; exit 1; }

"$EVAL_DIR/../venv/bin/python3" - "$CURRENT" "$BASELINE" "$TOL_PP" <<'PY'
import json, sys
cur, base, tol_pp = sys.argv[1:]
cur_j  = json.loads(open(cur).read())
base_j = json.loads(open(base).read())
tol = float(tol_pp) / 100.0
metrics = ("mrr", "hit@1", "hit@3", "hit@5")
regressions = []
print(f"\nDelta vs baseline (tolerance ±{tol_pp}pp):")
for m in metrics:
    delta = cur_j[m] - base_j[m]
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "·")
    flag  = "  ⚠ REGRESSION" if delta < -tol else ""
    print(f"  {m:<8} {base_j[m]:.3f} → {cur_j[m]:.3f}  {arrow}{abs(delta):+.3f}{flag}")
    if delta < -tol:
        regressions.append(m)
if regressions:
    print(f"\nRegressed: {', '.join(regressions)} — investigate before shipping retrieval changes.")
    sys.exit(1)
print("\n✓ within tolerance.")
PY
