import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import solution as s

def test_extract():
    assert s.extract_answer("<think>x</think> so \\boxed{1,024}.") == "1024"
    assert s.extract_answer("blah \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert s.extract_answer("<think>still thinking 42") is None
    assert s.extract_answer("</think>The final answer is 17.") == "17"
    assert s.extract_answer("\\boxed{17.0}") == "17"

def test_decide():
    assert s.decide(["5", "5", "5"]) is True
    assert s.decide(["5", "6", "5"]) is False
    assert s.decide([None, None]) is False
    assert s.decide(["5", None]) is False

def test_variants_keep_math():
    p = "Alice picks $x^2+3x$ integers between 1 and 100. Find the remainder modulo 1000."
    v = s.variants(p)
    assert v[0] == p and len(v) == 4
    for t in v:
        assert "$x^2+3x$" in t and "1000" in t
    assert "Mira" in v[3]

def test_budget_fallback(monkeypatch):
    monkeypatch.setattr(s, "_TOTAL_BUDGET_S", 0.0)
    out = s.are_robust("m", "low", ["a", "b"])
    assert out == [False, False] and all(type(x) is bool for x in out)

def test_generation_failure_keeps_coverage(monkeypatch):
    monkeypatch.setattr(s, "_TOTAL_BUDGET_S", 10_000.0)
    def boom(*a, **k): raise RuntimeError("oom")
    monkeypatch.setattr(s, "_generate_all", boom)
    assert s.are_robust("m", "low", ["a"]) == [False]

def test_mock_generation(monkeypatch):
    monkeypatch.setattr(s, "_TOTAL_BUDGET_S", 10_000.0)
    def fake(model_id, effort, prompts, budget):
        return ["</think>\\boxed{7}" if "Bob" not in p else "</think>\\boxed{8}" for p in prompts]
    monkeypatch.setattr(s, "_generate_all", fake)
    assert s.are_robust("m", "low", ["Find 3+4.", "Bob has 3+4 apples."]) == [True, False]
