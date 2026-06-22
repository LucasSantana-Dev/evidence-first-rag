# ADR-0010: Reranker model — portability vs. quality tradeoff

## Context

Once selective code-scope reranking is selected as the reranking strategy (ADR-0011), the next
decision is which cross-encoder reranker model to use by default. The tradeoff is between
model size/latency (portability) and ranking quality (effectiveness).

Two candidates were measured on the 50-case golden set:

1. **ms-marco-MiniLM-L-6-v2** — 88MB, small, fast
2. **BAAI/bge-reranker-v2-m3** — 2.1GB, large, slower but stronger

Both models were measured in code-scope reranking mode (ADR-0011) to understand their
effectiveness specifically for code retrieval, not just raw reranker strength.

## Measured baseline (50-case, code-scope reranking comparison, 2026-06-15)

### ms-marco-MiniLM-L-6-v2
```
  Latency per query: ~48ms
  Model size: 88MB
  Code-scope results (Hit@1 / MRR):
    No rerank baseline: 0.529 / 0.725
    With ms-marco rerank: 0.557 / 0.696
    Delta: +0.028 / −0.029 (marginal, slightly negative on MRR)
  Infrastructure scope (Hit@5 reference):
    No rerank baseline: 1.0
    With ms-marco: 0.75
    Delta: −0.25 (noticeable regression on infrastructure)
```

### BAAI/bge-reranker-v2-m3
```
  Latency per query: ~88ms (1.8× slower)
  Model size: 2.1GB
  Code-scope results (Hit@1 / MRR):
    No rerank baseline: 0.529 / 0.725
    With bge-v2-m3 rerank: 0.60 / 0.774
    Delta: +7.1pp / +4.9pp (strong improvement)
  Infrastructure scope (Hit@5 reference):
    No rerank baseline: 1.0
    With bge-v2-m3: 1.0
    Delta: 0.0 (no regression)
```

## Decision

**Default reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` for portability.**
**Optional upgrade: `BAAI/bge-reranker-v2-m3` when disk space and latency are acceptable.**

### Rationale

The bge-reranker-v2-m3 model is **strictly better** on measured quality:
- Code-scope improvement: +7.1pp Hit@1 / +4.9pp MRR
- No regression on non-code scopes (infrastructure Hit@5 stays at 1.0)
- Latency cost is 1.8× (+40ms per query), acceptable for offline/batch use

However, **portability is a non-negotiable property of a reusable tool:**
- The default reranker model must work on any machine with ~200MB free disk (model + dependencies).
- 2.1GB is a material barrier for local/edge deployment, CI runners, and resource-constrained environments.
- The codebase should not *require* users to download a 2.1GB model; they should be able to run
  the full harness with just the core 3 dependencies.

**Solution:** Ship ms-marco as the default (`RERANK_MODEL_DEFAULT`), then make bge-reranker-v2-m3
**opt-in via environment variable** (`RAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3`). This preserves
portability for the 90% case while enabling users to trade disk/latency for quality when they have
the budget.

### Documentation

Users who want the quality upgrade see the trade-off clearly in `ragcore/retrieval.py`:

```python
# Reranker model — two measured options (Pareto table, ROADMAP #5):
#   ms-marco-MiniLM-L-6-v2:  88MB,  48ms/query, Hit@1=0.529, MRR=0.696, infra Hit@5=0.75
#   BAAI/bge-reranker-v2-m3: 2.1GB, 88ms/query, Hit@1=0.647, MRR=0.767, infra Hit@5=1.0
# bge-v2-m3 is strictly better on quality (no regressions) at 1.8x latency.
# Default stays ms-marco-L6 for portability (88MB vs 2.1GB); set
# RAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3 when disk space allows.
```

## Consequences

- `RERANK_MODEL_DEFAULT = "cross-encoder/ms-marco-MiniLM-L-6-v2"` in `ragcore/retrieval.py`.
- When users want the better quality, they override: `RAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3`.
- The eval gate (`eval/check.sh`) uses the default (ms-marco) for reproducibility on any machine.
- Users who combine `RAG_CODE_RERANK=on` (selective code reranking) with the ms-marco default get
  modest quality gains on code; those with the bge upgrade get stronger gains.
- Graceful fallback (ADR-0011) ensures that if the model is unavailable or fails to load, queries
  still complete on the fused ranking.

## Revisit when

- A third reranker model becomes available with better portability/quality ratio (e.g., <500MB,
  >0.70 MRR on code) — benchmark and consider re-defaulting.
- Real usage telemetry shows that the majority of users set `RAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3`
  — at that point consider flipping the default (requires documenting the disk requirement more
  prominently and updating CI/packaging).
- A machine-learning community release (e.g., a code-specialized reranker) improves on bge-v2-m3 —
  benchmark it against the current Pareto frontier and update this decision.
- Disk/inference capacity improvements make the 2.1GB model negligible — revisit if average consumer
  hardware or CI budgets shift substantially.

## Related

- ADR-0011: Selective code-scope reranking (decides WHAT to rerank; this ADR decides WHICH model)
- `ragcore/retrieval.py`: Model initialization and Pareto table (lines 38–46)
- CHANGELOG (2026-06-17): Reranker Pareto table measurement results
- `eval/check.sh`: Uses default model for reproducible baseline
