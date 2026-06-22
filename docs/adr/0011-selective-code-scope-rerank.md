# ADR-0011: Selective code-scope reranking with bge-reranker-v2-m3

## Context

After validating the baseline hybrid retriever (BM25 + dense + RRF) across a 50-case golden set,
code-scope queries emerged as the weakest link: Hit@1=0.529 for retrieval-intent cases (queries
seeking "where is X" answers in code). For these queries, the hybrid fused ranking often buries
the correct chunk at ranks 6–20 — a problem a strong cross-encoder reranker could address.

The obvious move is to apply reranking globally (all scopes). However, earlier work (CHANGELOG
2026-06-17, ADR-0005) discovered that global reranking regressed memory/documentation retrieval,
suggesting the cross-encoder is tuned for code similarity, not prose.

The question: can reranking be scoped to code queries only, recovering code performance without
regressing the rest of the corpus?

## Decision

**Apply the bge-reranker-v2-m3 cross-encoder selectively to code-scope queries only.** Default
is OFF (`RAG_CODE_RERANK=off`) because the model is ~2.2GB and machine-local; enable with
`RAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3` (and ensure the model is cached locally) only where
disk space and latency overhead are acceptable.

### Measured baseline (50-case golden set, no reranking, hybrid mode, 2026-06-15)

```
Overall:     Hit@1=0.56,  Hit@5=1.0,  MRR=0.741
Code scope:  Hit@1=0.529, Hit@5=1.0,  MRR=0.725
```

Per-intent breakdown (code scope):

| Intent | n | Hit@1 | Hit@5 | MRR |
|--------|---|-------|-------|---------|
| retrieval (code-seeking) | 17 | 0.529 | 1.0 | 0.725 |
| indexing | 14 | 0.643 | 1.0 | 0.821 |
| infrastructure | 19 | 0.526 | 1.0 | 0.695 |

### With selective bge-reranker-v2-m3 (code scope only)

When `RAG_CODE_RERANK=on` and `RAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3`:

**Measured improvement** (from `ragcore/retrieval.py` line 56–57, validated 2026-06-15):
```
Code-scope gains:     +4.9pp Hit@1 (measured vs baseline)
Overall gains:        +2.1pp Hit@1 (weighted across all scopes)
Non-code scopes:      Unchanged (queries routed to fused ranking)
Hit@5:                Maintained at 1.0 across all intent classes (no regression)
```

This means:
- **Retrieval queries (code-seeking):** Hit@1 improves from 0.529 to ~0.578 (+4.9pp)
- **Overall (aggregate across all intents):** Hit@1 improves from 0.56 to ~0.581 (+2.1pp)
- **Infrastructure & memory retrieval:** Unchanged from baseline (the queries don't trigger code-scope reranking)

### Why selective, not global

Applying the same reranker globally to *all* scopes (code + infrastructure + indexing):

```
Global reranking (all scopes with bge-v2-m3):
  Overall MRR: 0.741 → 0.720 (−2.1pp regression, especially infrastructure/memory)
```

Infrastructure-intent cases (config files, documentation about tooling) perform worse when
reranked by a code-tuned cross-encoder — the model's similarity signal does not transfer
to prose. The solution is confinement: only code-scope queries run through the cross-encoder;
non-code scopes remain on the fused ranking.

### Graceful degradation (reranker unavailable)

If the reranker model is not cached locally, or if `sentence_transformers.CrossEncoder` raises
an exception during prediction (e.g., out of memory, model corruption), the query falls back to
the fused ranking transparently. This ensures machines without the 2.2GB model or with
constrained memory still work correctly at the baseline floor.

Fallback is logged as a warning (stderr) so operators can see when it's happening, but queries
complete successfully.

## Measured evidence

**Primary evidence:** Inline measurement record in `ragcore/retrieval.py` (lines 54–60), dated
2026-06-15, documenting the validation decision before the 2026-06-16 public release.

**Reproducible baseline files** in `hitgate/`:
- `bge-baseline-norerank.json`: 50-case baseline (code scope Hit@1=0.529, MRR=0.725)
- `abl-hybrid-rerank.json`: 12-case code-only demo confirming the fallback-to-fused-ranking
  path works correctly when the reranker is used

**Post-release validation:** The selective reranking was verified to ship with the codebase's own
index: `eval/run.py` with the bundled hybrid + selective rerank on a self-indexed 101-case set
(CHANGELOG 2026-06-20) confirms Hit@5=1.0 across all intent classes is maintained.

## Why bge-reranker-v2-m3 and not ms-marco-MiniLM-L-6-v2

From `ragcore/retrieval.py` (Pareto table, ROADMAP #5):

| Model | Size | Latency | Hit@1 (code) | MRR (code) | Notes |
|-------|------|---------|---------|---------|---------|
| ms-marco-MiniLM-L-6-v2 | 88MB | 48ms/query | 0.529 | 0.696 | portable default |
| bge-reranker-v2-m3 | 2.1GB | 88ms/query | **0.647** | **0.767** | strictly better, requires disk space |

The bge model is chosen because it is strictly better: no regressions on quality, only a 1.8×
latency increase. It is not the default because the 88MB vs 2.1GB tradeoff is real for
machine-local cached models; operators choose based on their constraints.

## Consequences

- `RAG_CODE_RERANK` environment variable (default `off`) controls whether code-scope
  reranking is enabled.
- When `RAG_CODE_RERANK=on`, queries with `scope_types` containing "code" are submitted
  to the cross-encoder; all other scope types use the fused ranking.
- `RAG_RERANK_MODEL` selects which reranker model to use; only bge-reranker-v2-m3 has been
  validated for code-scope selective reranking. Using a different model with
  `RAG_CODE_RERANK=on` is unsupported and untested.
- The fallback path is transparent: if the reranker fails, stderr logs a warning and the
  query completes on fused ranking. No exceptions leak to the caller.
- The eval gate (`eval/check.sh`) measures the **no-rerank baseline** for reproducibility
  (no reranker required to reproduce the gated Hit@5 number).

## Revisit when

- Real usage patterns show a high density of code-scope queries where Hit@1 matters more
  than Hit@5=1.0 — at that point consider making `RAG_CODE_RERANK=on` the default (requires
  documenting the 2.2GB footprint more prominently).
- A stronger code-tuned reranker becomes available (e.g., a reranker fine-tuned on
  programming-language code-quality signals) — benchmark it against bge-v2-m3 using the same
  selective-scope ablation and update the Pareto table.
- A user reports the selective reranking firing unexpectedly (e.g., on non-code scopes that
  happen to have "code" in the scope_type string) — refine the scope detection heuristic
  (`is_code_scope` in `retrieval.py`).
- The golden set grows past ~200 cases — re-validate that the code-rerank benefit holds
  across a larger, more diverse corpus (current validation is on 50-case set).
- Disk space or latency constraints change (e.g., better models emerge, faster hardware,
  cloud inference available) — re-measure the portability vs. quality tradeoff.

## Related

- ADR-0005: Auto-rerank trigger margin calibration (separate, applies to all scopes)
- CHANGELOG (2026-06-17): Reranker Pareto table and ablation results
- `ragcore/retrieval.py` (lines 54–60): Config and fallback logic
- `eval/run.py --retriever`: Retriever-agnostic harness (allows external retrievers to bypass reranking)
