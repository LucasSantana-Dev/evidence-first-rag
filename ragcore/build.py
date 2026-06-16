#!/usr/bin/env python3
"""Chunk + embed curated sources into ~/.claude/rag-index/index.sqlite.

Markdown sources (memory, plans, handoffs, skills, codex, changelog, repo-docs,
repo-readme) plus curated source-code dirs (src/**) for a whitelisted set of
repos. Language-aware chunkers split by symbol where possible.

Usage:
  build.py                                           # full rebuild
  build.py --incremental <file> [...files]           # reindex specific files
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import subprocess
from chunkers import chunk_file, detect_language

ROOT = Path.home() / ".claude" / "rag-index"
DB = ROOT / "index.sqlite"
MODEL_NAME = "intfloat/multilingual-e5-small"
DIM = 384
HOME = Path.home()
MAX_FILE_BYTES = 200_000
EXCLUDED_DIR_PARTS = {
    "site-packages",
    "htmlcov",
    ".eggs",
    ".tox",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "coverage",
    ".git",
    ".next",
    ".turbo",
    "venv",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "test-results",
    "playwright-report",
    ".storybook",
    ".docusaurus",
    ".worktrees",
    "worktrees",
    ".wt-res",
    ".wt-luckynotify",
    ".wt-notify",
    ".wt-ufw",
    ".wt-strict",
    ".wt-renovate",
    ".wt-disc",
    ".wt-fix",
    ".wt-notify-clean",
}
CODE_EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".sh", ".bash", ".zsh"}

CURATED_REPOS = [
    HOME / "Desenvolvimento" / "Lucky",
    HOME / "Desenvolvimento" / "homelab",
    HOME / "Desenvolvimento" / "Craftvaria",
    HOME / "Desenvolvimento" / "ai-dev-toolkit",
    HOME / "Desenvolvimento" / "ai-dev-toolkit-setup",
    # Forge-Space repos (core, siza-gen, ui-mcp, mcp-gateway, branding-mcp,
    # siza-desktop) removed 2026-05-03 — org wound down 2026-04-20 per
    # MEMORY.md; their code was generating noise in RAG queries.
]

WORKSTATION_CODE_GLOBS = [
    str(HOME / ".claude/rag-index/*.py"),
    str(HOME / ".claude/rag-index/*.sh"),
    str(HOME / ".codex/scripts/*.sh"),
]

SOURCES: list[tuple[str, str]] = [
    ("memory", str(HOME / ".claude/projects/*/memory/*.md")),
    ("plans", str(HOME / ".claude/plans/*.md")),
    ("adrs", str(HOME / ".claude/adrs/*.md")),
    ("handoffs", str(HOME / ".claude/handoffs/*/*.md")),
    ("skills", str(HOME / ".claude/skills/*/SKILL.md")),
    ("standards", str(HOME / ".claude/standards/*.md")),
    ("codex", str(HOME / ".codex/AGENTS.md")),
    ("codex-rules", str(HOME / ".codex/rules/*.rules")),
    ("codex-memory", str(HOME / ".codex/memories/**/*.md")),
    ("serena", str(HOME / ".serena/memories/**/*.md")),
    ("memory-export", str(HOME / ".claude/memory-exports/*.md")),
]
for repo in CURATED_REPOS:
    if not repo.is_dir():
        continue
    SOURCES.append(("changelog", str(repo / "CHANGELOG.md")))
    SOURCES.append(("repo-docs", str(repo / "docs/**/*.md")))
    SOURCES.append(("repo-readme", str(repo / "README.md")))
    SOURCES.append(("spec", str(repo / "docs/specs/**/*.md")))
    SOURCES.append(("roadmap", str(repo / "docs/roadmap.md")))
    SOURCES.append(("serena", str(repo / ".serena/memories/*.md")))
for glob in WORKSTATION_CODE_GLOBS:
    SOURCES.append(("workstation-code", glob))


SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    repo TEXT,
    language TEXT,
    symbol TEXT,
    path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL,
    file_sha TEXT NOT NULL,
    mtime REAL NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE INDEX IF NOT EXISTS chunks_type ON chunks(source_type);
CREATE INDEX IF NOT EXISTS chunks_repo ON chunks(repo);
"""


def iter_md_sources() -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []
    for stype, glob in SOURCES:
        base = Path(glob.split("*", 1)[0])
        if "**" in glob:
            pattern = glob.split("/**/", 1)[1]
            for p in base.rglob(pattern):
                if p.is_file():
                    results.append((stype, p))
        elif "*" in glob:
            parent = Path(glob).parent
            name_pat = Path(glob).name
            if "*" in str(parent):
                # Handle double-wildcard patterns like .../*/memory/*.md
                # Extract everything after the first wildcard (e.g., /memory/*.md)
                grandparent = Path(str(parent).split("*", 1)[0])
                rest_pattern = glob.split("*", 1)[1].lstrip("/")
                for sub in grandparent.glob("*"):
                    if sub.is_dir():
                        for p in sub.glob(rest_pattern):
                            if p.is_file():
                                results.append((stype, p))
            else:
                for p in parent.glob(name_pat):
                    if p.is_file():
                        results.append((stype, p))
        else:
            p = Path(glob)
            if p.is_file():
                results.append((stype, p))
    return results


GIT_LOG_DAYS = 180


def collect_commit_chunks() -> list[dict]:
    """Run git log on each curated repo; one chunk per commit (subject + body head)."""
    rows: list[dict] = []
    for repo in CURATED_REPOS:
        if not (repo / ".git").exists():
            continue
        try:
            proc = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "log",
                    f"--since={GIT_LOG_DAYS}.days",
                    "--no-merges",
                    "--pretty=format:%H%x01%ai%x01%an%x01%s%x01%b%x02",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            continue
        raw = proc.stdout
        for entry in raw.split("\x02"):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split("\x01", 4)
            if len(parts) < 4:
                continue
            sha, date_str, author, subject = parts[:4]
            body = parts[4] if len(parts) == 5 else ""
            body_head = body.strip().splitlines()
            body_top = "\n".join(body_head[:6])[:1200]
            text = f"{subject}\n\n{body_top}".strip()
            rows.append(
                {
                    "source_type": "commit",
                    "repo": repo.name,
                    "language": "git",
                    "symbol": sha[:7],
                    "path": f"git:{repo.name}@{sha}",
                    "start": 0,
                    "end": 0,
                    "text": text[:4000],
                    "sha": sha,
                    "mtime": 0.0,
                    "meta": f"{author} · {date_str}",
                }
            )
    return rows


def iter_code_sources() -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []
    for repo in CURATED_REPOS:
        if not repo.is_dir():
            continue
        for p in repo.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in CODE_EXTS:
                continue
            if is_excluded_path(p.relative_to(repo)):
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            results.append(("code", p))
    return results


def is_excluded_path(relative_path: Path) -> bool:
    for part in relative_path.parts:
        if part in EXCLUDED_DIR_PARTS:
            return True
        if part.startswith(".venv"):   # .venv_ci, .venv_dev, etc.
            return True
        if part.startswith(".wt-"):
            return True
    return False


def file_sha(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def classify_repo(path: Path) -> str | None:
    # Compare against both the symlink path (~/Desenvolvimento/X) and its
    # resolved target (/Volumes/External HD/...) — resolve() expands the
    # symlink, so matching only the unresolved repo path never hits.
    rp = path.resolve()
    for repo in CURATED_REPOS:
        for base in (repo, repo.resolve()):
            try:
                rp.relative_to(base)
                return repo.name
            except ValueError:
                continue
    return None


def classify_type(path: Path) -> str:
    s = str(path)
    if "/memory/" in s:
        return "memory"
    if "/plans/" in s:
        return "plans"
    if "/adrs/" in s and s.endswith(".md"):
        return "adrs"
    if "/handoffs/" in s:
        return "handoffs"
    if "/.claude/skills/" in s or "/.agents/skills/" in s:
        return "skills"
    if "/.codex/" in s:
        return "codex"
    if "/standards/" in s and s.endswith(".md"):
        return "standards"
    if s.endswith("CHANGELOG.md"):
        return "changelog"
    if s.endswith("README.md"):
        return "repo-readme"
    if s.endswith("/docs/roadmap.md") or s.endswith("docs/roadmap.md"):
        return "roadmap"
    if "/docs/specs/" in s and s.endswith(".md"):
        return "spec"
    if "/docs/" in s and s.endswith(".md"):
        return "repo-docs"
    if s.startswith(str(HOME / ".claude/rag-index/")) or s.startswith(str(HOME / ".codex/scripts/")):
        return "workstation-code"
    if path.suffix.lower() in CODE_EXTS:
        return "code"
    return "other"


def connect() -> sqlite3.Connection:
    ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA busy_timeout=10000")  # WAL+timeout hardening
    conn.executescript(SCHEMA)
    return conn


def embed(model, texts: list[str]) -> np.ndarray:
    # E5 model requires "passage: " prefix for indexed chunks; this is added
    # at embed time only, not stored in the database.
    prefixed_texts = [f"passage: {t}" for t in texts]
    vecs = model.encode(
        prefixed_texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    return vecs.astype(np.float32)


def index_files(
    conn: sqlite3.Connection, model, files: list[tuple[str, Path]], purge_paths: list[str]
) -> int:
    for p in purge_paths:
        conn.execute("DELETE FROM chunks WHERE path = ?", (p,))
    total = 0
    batch_texts: list[str] = []
    batch_meta: list[dict] = []
    for stype_hint, path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        sha = file_sha(path)
        mtime = path.stat().st_mtime
        stype = stype_hint if stype_hint != "code" else "code"
        repo = classify_repo(path)
        language = detect_language(path)
        for start, end, body, symbol in chunk_file(path, text):
            # Contextual chunk prefix for embedding: adds source type, repo, file name, symbol
            # Format: "<source_type> | <repo or ''> | <file name or doc title> | <symbol or ''>\n<chunk body>"
            # This is only used for embedding, not stored in the database.
            file_name = path.name
            context_prefix = f"{stype} | {repo or ''} | {file_name} | {symbol or ''}"
            contextualized_text = f"{context_prefix}\n{body[:4000]}"

            batch_texts.append(contextualized_text)
            batch_meta.append(
                {
                    "source_type": stype,
                    "repo": repo,
                    "language": language,
                    "symbol": symbol,
                    "path": str(path),
                    "start": start,
                    "end": end,
                    "text": body[:4000],
                    "sha": sha,
                    "mtime": mtime,
                }
            )
            if len(batch_texts) >= 64:
                _flush(conn, model, batch_texts, batch_meta)
                total += len(batch_texts)
                batch_texts.clear()
                batch_meta.clear()
    if batch_texts:
        _flush(conn, model, batch_texts, batch_meta)
        total += len(batch_texts)
    conn.commit()
    return total


def _flush(conn, model, texts, meta):
    vecs = embed(model, texts)
    rows = []
    for m, vec in zip(meta, vecs):
        rows.append(
            (
                m["source_type"],
                m["repo"],
                m["language"],
                m["symbol"],
                m["path"],
                m["start"],
                m["end"],
                m["text"],
                m["sha"],
                m["mtime"],
                vec.tobytes(),
            )
        )
    conn.executemany(
        "INSERT INTO chunks (source_type, repo, language, symbol, path, start_line, end_line, text, file_sha, mtime, embedding) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incremental", nargs="+", help="specific files to reindex")
    ap.add_argument("--no-code", action="store_true", help="skip source code ingestion")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    conn = connect()
    started = time.time()

    if args.incremental:
        targets: list[tuple[str, Path]] = []
        purge: list[str] = []
        for raw in args.incremental:
            p = Path(raw).expanduser().resolve()
            purge.append(str(p))
            if not p.exists():
                continue
            stype = classify_type(p)
            targets.append((stype, p))
        written = index_files(conn, model, targets, purge)
        print(f"incremental: {len(targets)} files, {written} chunks, {time.time()-started:.1f}s")
    else:
        conn.execute("DELETE FROM chunks")
        md_files = iter_md_sources()
        code_files = [] if args.no_code else iter_code_sources()
        files = md_files + code_files
        written = index_files(conn, model, files, [])
        commits_written = 0
        if not args.no_code:
            commit_rows = collect_commit_chunks()
            if commit_rows:
                texts = [r["text"] for r in commit_rows]
                # Embed in batches to share with index_files pathway
                for i in range(0, len(commit_rows), 64):
                    batch = commit_rows[i : i + 64]
                    # Add contextual prefix for commits: "commit | <repo>"
                    contextualized_texts = [
                        f"commit | {r['repo']}\n{r['text']}" for r in batch
                    ]
                    vecs = embed(model, contextualized_texts)
                    sql_rows = []
                    for m, vec in zip(batch, vecs):
                        sql_rows.append(
                            (
                                m["source_type"],
                                m["repo"],
                                m["language"],
                                m["symbol"],
                                m["path"],
                                m["start"],
                                m["end"],
                                m["text"],
                                m["sha"],
                                m["mtime"],
                                vec.tobytes(),
                            )
                        )
                    conn.executemany(
                        "INSERT INTO chunks (source_type, repo, language, symbol, path, start_line, end_line, text, file_sha, mtime, embedding) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        sql_rows,
                    )
                    commits_written += len(batch)
                conn.commit()
        print(
            f"full rebuild: md={len(md_files)} code={len(code_files)} commits={commits_written} chunks={written+commits_written} t={time.time()-started:.1f}s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
