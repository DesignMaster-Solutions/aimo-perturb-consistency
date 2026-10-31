# Reproduce the public-sample audit

This is a descriptive audit of the organizers' public development sample, not training data for our submission and not the private validation predictions.

```sh
curl -fL https://huggingface.co/datasets/aimo-interp/val-sample/resolve/1ae454ec1fad9727084eda8f9f3c9ae2239b21de/data/validation-00000-of-00001.parquet -o /tmp/aimo-public.parquet
uv run --no-project --with pandas,pyarrow python3 report/analysis/public_sample.py /tmp/aimo-public.parquet
```

The script rejects a different file hash. The committed `public_sample.json` is the complete output. No model inference or paid API is used.

There are 28 rows, 24 distinct `(model_id, dataset_id, problem_id)` pairs, eight problems and seven model IDs. Four rows repeat a model/problem pair; no row is an exact duplicate. Robust labels agree within each repeated pair. Counting each pair once changes the always-false descriptive baseline from 19/28 (67.86%) to 15/24 (62.50%). This does not change any Codabench score.

The report previously inferred identical predictions from equal aggregate ablation accuracies. That inference was unsupported and has been removed. Model groups also cover different problem subsets: model-level label means do not isolate a model effect. The exact labeling threshold is unpublished, so the report no longer claims that losing one sampled correct answer determines the label.

Official rules checked on 2026-10-31: https://aimo-interp.github.io/ . Writeup prizes total USD 4,000 across selected reports; no individual award amount is specified. Reports are due November 5 AoE, decisions November 15 AoE. The existing OpenReview activation watch remains necessary; a public PDF is not an OpenReview submission.
