# Decisions — how this system was steered

Most portfolios show what someone *built*. This one also shows what was deliberately
**not** built, and why — because on a single-user system with no users to satisfy,
judgment about where *not* to spend effort is the scarcest skill, and the easiest to
fake in an interview.

Every non-trivial change went through the same loop: **measure → challenge → decide →
record the trigger that would reopen it.** Three decisions are worth reading.

## The discipline, in one rule

> No change ships without a *measured* before/after, and no decision is recorded
> without the single condition that would **reopen** it. A decision with no revisit
> trigger is a guess wearing a timestamp.

## 1. Make the metric honest before trusting it

The retrieval eval was quietly lying: a chunk of its "golden" cases were *un-winnable*
— the answer wasn't in the corpus at all — so they capped the score with a constant
penalty that looked like a quality floor. Auditing and removing them moved the
baseline by ~8 percentage points. The standing rule that came out of it: **a benchmark
you haven't audited for contamination is worse than no benchmark**, because it attaches
a number to false confidence. Every downstream decision only became trustworthy once
the metric stopped lying. (You can see the discipline in the demo eval: the two failing
cases are left in, not deleted.)

## 2. Defer the machine learning — the system is already at its ceiling *(a deliberate "no")*

The obvious "make it smarter" move was to train a learned ranker or fine-tune the
embeddings. Researched, critiqued, and **declined**:

- A learned-to-rank model wants 10⁴–10⁵ labeled queries; a hand-curated eval has ~100.
  At that scale a trained ranker overfits and loses to the tuned baseline.
- The one ML lever genuinely worth having — a cross-encoder reranker — is *already*
  here, and scoped to where it was measured to help (code), not applied blindly.
- Manufacturing labels with an LLM is circular: candidates drawn from the current
  retriever bake in the current retriever's mistakes, so a model trained on them can
  never learn to promote what it's already burying.

So "add ML" resolved to: **don't train anything; improve *measurement* instead.** The
reopen trigger is explicit — accumulate ~500 *real* (not manufactured) relevance labels
from actual use, then re-run the feasibility check against a held-out set.

## 3. Index all code uniformly; keep the core vendor-neutral *(the other "no")*

This system was extracted from a personal AI-assistant memory index that was wired to
one assistant's session format and to an external AST code-graph tool. Both were
**removed** from the core rather than shipped:

- The code-graph integration excluded a repo's raw code in favor of a separately-built
  graph. Removing it means the core simply indexes all code uniformly — a strict
  *superset* of the graph-gated behavior, with one fewer external dependency.
- The assistant-transcript ingester coupled the core to a private, undocumented format.
  It belongs in an opt-in adapter, not the core.

Cutting features to gain portability is a harder call than adding them. The result is a
core that runs on any source tree with three pip dependencies, which is the entire
point of a reusable tool.

## Why a "no" is a portfolio asset

Shipping is the default; restraint is the differentiator. Each "no" above is backed by
a measurement, an adversarial review, and a written trigger that would change the
answer. That loop — measure, challenge, decide, leave a tripwire — is the actual
transferable skill this repository exists to demonstrate.
