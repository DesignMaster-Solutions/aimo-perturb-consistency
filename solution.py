"""Perturbation self-consistency: a black-box robustness check run at test time.

The task label says a model is robust on a problem when its answer survives
perturbations that keep the deductive chain intact (typos, renaming, distractors,
rephrasing). We measure exactly that, on the evaluated checkpoint itself: answer
the original problem and a few rule-based perturbations of it with greedy
decoding, and call the pair robust only if every answer agrees.

Anything we cannot measure inside the time budget (no final answer, out of
time, load failure) falls back to False, the majority class in every public
sample (robust share 0.32 to 0.375).
"""

from __future__ import annotations

import gc
import os
import re
import time

_START = time.monotonic()
# The 3600 s limit covers the whole prediction run; keep a margin for loading
# and scoring. Overridable for local tests.
_TOTAL_BUDGET_S = float(os.environ.get("PC_TOTAL_BUDGET_S", "3300"))
_EXPECTED_CALLS = int(os.environ.get("PC_EXPECTED_CALLS", "5"))
_MAX_NEW_TOKENS = int(os.environ.get("PC_MAX_NEW_TOKENS", "6144"))
_BATCH = int(os.environ.get("PC_BATCH", "24"))
_calls_done = 0

_TYPO_SKIP = re.compile(r"[\d$\\{}^_=+\-*/()<>|]")
_NAME_CANDIDATES = ("Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi")
_NAME_SWAP = ("Mira", "Tomas", "Ines", "Kofi", "Lena", "Ravi", "Sora", "Yuki")
_DISTRACTOR = (
    "Unrelated note: a nearby library has 37 shelves and opened in 1994; "
    "this does not affect the problem."
)


def perturb_typos(text: str) -> str:
    """Swap two inner letters in every 4th plain word; math tokens are untouched."""
    words = text.split(" ")
    out = []
    for i, w in enumerate(words):
        if i % 4 == 3 and len(w) >= 5 and w.isalpha() and not _TYPO_SKIP.search(w):
            w = w[:2] + w[3] + w[2] + w[4:]
        out.append(w)
    return " ".join(out)


def perturb_rename(text: str) -> str:
    """Rename common character names (no-op when none appear)."""
    renamed = text
    for old, new in zip(_NAME_CANDIDATES, _NAME_SWAP):
        renamed = re.sub(rf"\b{old}\b", new, renamed)
    return renamed


def perturb_distract(text: str) -> str:
    return f"{text}\n\n{_DISTRACTOR}"


def variants(problem: str) -> list[str]:
    out = [problem, perturb_typos(problem), perturb_distract(problem)]
    renamed = perturb_rename(problem)
    if renamed != problem:
        out.append(renamed)
    return out


_BOXED = re.compile(r"\\boxed\s*\{")
_INT = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _last_boxed(text: str) -> str | None:
    start = None
    for m in _BOXED.finditer(text):
        start = m.end()
    if start is None:
        return None
    depth, i = 1, start
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[start : i - 1] if depth == 0 else None


def extract_answer(text: str) -> str | None:
    """Final answer: last \\boxed{...}, else the last number after the reasoning ends."""
    tail = text.split("</think>")[-1]
    boxed = _last_boxed(tail) or _last_boxed(text)
    if boxed is not None:
        cand = boxed
    else:
        if "</think>" not in text and "final" not in tail.lower():
            return None  # reasoning was cut before an answer
        nums = _INT.findall(tail)
        if not nums:
            return None
        cand = nums[-1]
    cand = cand.replace(",", "").replace(" ", "").strip().rstrip(".")
    cand = re.sub(r"^\\text\{(.*)\}$", r"\1", cand)
    try:
        f = float(cand)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return cand or None


def decide(answers: list[str | None]) -> bool:
    """Robust only if the original has an answer and every perturbation matches it."""
    if not answers or answers[0] is None:
        return False
    return all(a == answers[0] for a in answers[1:])


def _budget_for_this_call() -> float:
    remaining = _TOTAL_BUDGET_S - (time.monotonic() - _START)
    calls_left = max(1, _EXPECTED_CALLS - _calls_done)
    return remaining / calls_left


def _format(tokenizer, text: str, reasoning_effort: str) -> str:
    msgs = [{"role": "user", "content": text + "\n\nPut the final answer in \\boxed{}."}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if reasoning_effort in ("low", "medium", "high"):
        kwargs["reasoning_effort"] = reasoning_effort
    try:
        return tokenizer.apply_chat_template(msgs, **kwargs)
    except Exception:  # noqa: BLE001 - template variations across checkpoints
        try:
            kwargs.pop("reasoning_effort", None)
            return tokenizer.apply_chat_template(msgs, **kwargs)
        except Exception:  # noqa: BLE001
            return msgs[0]["content"]


def _generate_all(model_id: str, reasoning_effort: str, prompts: list[str], budget_s: float) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    deadline = time.monotonic() + budget_s
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, device_map="auto", dtype="auto"
    )
    model.eval()
    device = model.get_input_embeddings().weight.device
    texts: list[str] = [""] * len(prompts)
    try:
        # Longest prompts first keeps padding small inside each batch.
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        for s in range(0, len(order), _BATCH):
            left = deadline - time.monotonic()
            if left < 20:
                break
            idx = order[s : s + _BATCH]
            batches_left = (len(order) - s + _BATCH - 1) // _BATCH
            enc = tok([_format(tok, prompts[i], reasoning_effort) for i in idx],
                      return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=_MAX_NEW_TOKENS,
                    do_sample=False,
                    max_time=max(10.0, left / batches_left),
                    pad_token_id=tok.pad_token_id,
                )
            gen = out[:, enc["input_ids"].shape[1]:]
            for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=False)):
                texts[i] = t
            del enc, out, gen
    finally:
        del model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return texts


def are_robust(model_id: str, reasoning_effort: str = "default", problems: list[str] | None = None) -> list[bool]:
    global _calls_done
    problems = list(problems or [])
    if not problems:
        return []
    budget = _budget_for_this_call()
    _calls_done += 1
    if budget < 60:
        return [False] * len(problems)
    groups = [variants(p) for p in problems]
    flat = [v for g in groups for v in g]
    try:
        texts = _generate_all(model_id, reasoning_effort, flat, budget)
    except Exception:  # noqa: BLE001 - never lose coverage on a failed batch
        return [False] * len(problems)
    preds, k = [], 0
    for g in groups:
        answers = [extract_answer(t) if t else None for t in texts[k : k + len(g)]]
        preds.append(bool(decide(answers)))
        k += len(g)
    return preds
