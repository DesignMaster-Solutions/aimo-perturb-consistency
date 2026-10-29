# Perturbation self-consistency for the AIMO Interpretability Challenge

Submission by Design Master Solutions LLC to the [AIMO Interpretability Challenge](https://aimo-interp.github.io/) (NeurIPS 2026), Small Model and Main tracks.

## Idea

The challenge label calls a model robust on a problem when its answer survives perturbations that leave the deductive chain unchanged (typos, renaming, distractors, rephrasing). Instead of learning a proxy for that label from a 28-row public sample, this method measures it directly at test time on the checkpoint being judged:

1. Build rule-based perturbations of each problem that never touch math tokens: letter-swap typos in plain words, an irrelevant distractor sentence, and character renaming when names appear.
2. Answer the original and every perturbation with greedy decoding on the evaluated model (offline, `local_files_only=True`).
3. Extract the final answer (last `\boxed{}`, else the last number after the reasoning ends).
4. Predict robust only if the original produced an answer and every perturbation agrees with it.

Anything that cannot be measured inside the 3600 s budget (no final answer, time out, load failure) falls back to `False`, the majority class in every public sample (robust share 0.32 to 0.375). A global time budget is split across model calls and passed to `generate(max_time=...)`, so coverage stays at 1.0.

It is a black-box method: no weights, activations or probes. Known limits: domain and rephrase perturbations need a generator model and are not reproduced; a perturbation set this small can call a brittle model robust when it happens to be stable on these three edits.

## Files

- `solution.py`: the whole submission (`are_robust(model_id, reasoning_effort, problems)`).
- `tests/test_pc.py`: answer extraction, decision rule, perturbations keep math intact, budget and failure fallbacks (`uvx pytest -q tests`).

Build the bundle: `zip pc-main.zip solution.py` (add an empty `small.txt` for the Small Model track).

## License

MIT
