"""The harness is retriever-agnostic. These assert the plug-in contract and the metric
math directly, with a stub retriever — no model, no index, no built-in coupling."""
import pytest

from hitgate.run import load_retriever, run


def test_run_is_retriever_agnostic_and_computes_metrics():
    # stub retriever: q1 answer at rank 1, q2 at rank 3, q3 missing entirely.
    canned = {
        "q1": [{"path": "src/right.py"}, {"path": "x.py"}],
        "q2": [{"path": "x.py"}, {"path": "y.py"}, {"path": "src/right.py"}],
        "q3": [{"path": "x.py"}],
    }
    stub = lambda query, top, scope: canned[query][:top]
    cases = [
        {"query": "q1", "expect_path_contains": "right.py", "expect_scope": "code"},
        {"query": "q2", "expect_path_contains": "right.py", "expect_scope": "code"},
        {"query": "q3", "expect_path_contains": "right.py", "expect_scope": "code"},
    ]
    out = run(cases, top=5, retriever=stub)
    assert out["n"] == 3
    assert out["hit@1"] == round(1 / 3, 3)      # only q1 at rank 1
    assert out["hit@3"] == round(2 / 3, 3)      # q1 + q2
    assert out["hit@5"] == round(2 / 3, 3)      # q3 never returns the answer
    assert out["mrr"] == round((1 / 1 + 1 / 3 + 0) / 3, 3)


def test_default_loads_builtin():
    r = load_retriever(None)
    assert callable(r)


def test_spec_imports_external_callable():
    r = load_retriever("hitgate.example_external_retriever:retrieve")
    assert callable(r)


def test_malformed_spec_exits():
    with pytest.raises(SystemExit):
        load_retriever("not-a-valid-spec")  # no ':'


def test_unknown_module_exits():
    with pytest.raises(SystemExit):
        load_retriever("nope.does.not.exist:retrieve")


def test_example_retriever_returns_protocol_shape(tmp_path, monkeypatch):
    (tmp_path / "fusion.py").write_text("def reciprocal_rank_fusion():\n    return None\n")
    (tmp_path / "other.py").write_text("def unrelated():\n    return None\n")
    monkeypatch.setenv("RAG_SOURCE_ROOTS", str(tmp_path))
    from hitgate.example_external_retriever import retrieve

    res = retrieve("reciprocal rank fusion", top=3, scope="code")
    assert res and all("path" in r for r in res)
    assert "fusion.py" in res[0]["path"]  # the file with the matching terms ranks first


def test_run_validates_retriever_result_has_path_key():
    """Retriever returning result without 'path' key should raise clear ValueError."""
    stub = lambda query, top, scope: [{"content": "missing path key"}]
    cases = [
        {"query": "q1", "expect_path_contains": "right.py", "expect_scope": "code"},
    ]
    with pytest.raises(ValueError) as exc_info:
        run(cases, top=5, retriever=stub)
    assert "path" in str(exc_info.value).lower()
    assert "retriever" in str(exc_info.value).lower()


def test_run_validates_retriever_result_path_is_string():
    """Retriever returning non-string 'path' value should raise clear TypeError."""
    stub = lambda query, top, scope: [{"path": None}]
    cases = [
        {"query": "q1", "expect_path_contains": "right.py", "expect_scope": "code"},
    ]
    with pytest.raises(TypeError) as exc_info:
        run(cases, top=5, retriever=stub)
    assert "path" in str(exc_info.value).lower()
    assert "str" in str(exc_info.value).lower()


def test_run_validates_retriever_result_path_is_string_with_int():
    """Retriever returning integer 'path' value should raise clear TypeError."""
    stub = lambda query, top, scope: [{"path": 42}]
    cases = [
        {"query": "q1", "expect_path_contains": "right.py", "expect_scope": "code"},
    ]
    with pytest.raises(TypeError) as exc_info:
        run(cases, top=5, retriever=stub)
    assert "path" in str(exc_info.value).lower()
    assert "str" in str(exc_info.value).lower()


def test_run_validates_retriever_result_is_dict():
    """Retriever returning non-dict result should raise clear TypeError."""
    stub = lambda query, top, scope: ["not a dict"]
    cases = [
        {"query": "q1", "expect_path_contains": "right.py", "expect_scope": "code"},
    ]
    with pytest.raises(TypeError) as exc_info:
        run(cases, top=5, retriever=stub)
    assert "mapping" in str(exc_info.value).lower() or "dict" in str(exc_info.value).lower()
    assert "retriever" in str(exc_info.value).lower()


def test_run_validates_with_valid_retriever_still_works():
    """Valid retriever with proper shape should continue to work (regression test)."""
    canned = {
        "q1": [{"path": "src/right.py"}, {"path": "x.py"}],
    }
    stub = lambda query, top, scope: canned[query][:top]
    cases = [
        {"query": "q1", "expect_path_contains": "right.py", "expect_scope": "code"},
    ]
    out = run(cases, top=5, retriever=stub)
    assert out["n"] == 1
    assert out["hit@1"] == 1.0
