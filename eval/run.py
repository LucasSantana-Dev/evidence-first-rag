#!/usr/bin/env python3
"""Evaluate retrieval quality against a curated Q/A dataset.

Inputs:  eval/golden.demo.jsonl  (one JSON per line: query, expect_path_contains, expect_scope)
Outputs: MRR, Hit@1, Hit@3, Hit@5 — prints table + writes eval/<label>.json

Usage:
  eval/run.py                         # scores the demo golden set
  eval/run.py --label post-reranker   # save as named run
  eval/run.py --top 10 --label wider
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ragcore"))
from retrieval import search

DATASET = ROOT / "eval" / "golden.demo.jsonl"


def load(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(cases: list[dict], top: int, rerank: bool = False) -> dict:
    per_case = []
    for case in cases:
        # Tolerate the legacy diff-scoped schema (id/type/expected_files) by skipping it.
        if "expect_path_contains" not in case:
            continue
        q = case["query"]
        expect = case["expect_path_contains"]
        scope = case.get("expect_scope")
        scope_key = scope if isinstance(scope, str) and scope else ("+".join(scope) if isinstance(scope, list) and scope else "none")
        expected = expect if isinstance(expect, list) else [expect]
        results = search(q, top=top, scope_types=scope, scope_repos=["all"], cwd=None, rerank=rerank)
        hit_rank = None
        for r in results:
            if any(e in r["path"] for e in expected):
                hit_rank = r["rank"]
                break
        per_case.append(
            {
                "query": q,
                "expect": expect,
                "scope": scope_key,
                "hit_rank": hit_rank,
                "top_hit": f"{results[0]['path']}:{results[0]['start_line']}" if results else None,
            }
        )

    def metrics(cases: list[dict]) -> dict:
        n = len(cases) or 1
        hits_at = lambda k: sum(1 for c in cases if c["hit_rank"] and c["hit_rank"] <= k) / n
        mrr = sum((1.0 / c["hit_rank"]) if c["hit_rank"] else 0.0 for c in cases) / n
        return {
            "n": len(cases),
            "mrr": round(mrr, 3),
            "hit@1": round(hits_at(1), 3),
            "hit@3": round(hits_at(3), 3),
            "hit@5": round(hits_at(5), 3),
        }

    by_scope: dict[str, list] = {}
    for c in per_case:
        by_scope.setdefault(c["scope"], []).append(c)

    out = metrics(per_case)
    out["by_scope"] = {sc: metrics(cs) for sc, cs in sorted(by_scope.items())}
    out["per_case"] = per_case
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--dataset", default=str(DATASET), help="path to eval dataset jsonl")
    ap.add_argument("--rerank", action="store_true", help="enable cross-encoder reranking")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cases = load(Path(args.dataset))
    started = time.time()
    result = run(cases, top=args.top, rerank=args.rerank)
    elapsed = time.time() - started

    rerank_tag = " [RERANK]" if args.rerank else " [FAST]"
    print(
        f"[{args.label}]{rerank_tag}  n={result['n']}  MRR={result['mrr']}  "
        f"hit@1={result['hit@1']}  hit@3={result['hit@3']}  hit@5={result['hit@5']}  "
        f"({elapsed:.1f}s)"
    )
    for sc, m in result.get("by_scope", {}).items():
        print(f"    scope={sc:10} n={m['n']:>3}  hit@1={m['hit@1']}  hit@5={m['hit@5']}  mrr={m['mrr']}")
    if args.verbose:
        for c in result["per_case"]:
            status = "✓" if c["hit_rank"] else "✗"
            rank = f"#{c['hit_rank']}" if c["hit_rank"] else "MISS"
            print(f"  {status} {rank:>4}  {c['query'][:60]}  → {c['top_hit']}")
    out_path = ROOT / "eval" / f"{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
