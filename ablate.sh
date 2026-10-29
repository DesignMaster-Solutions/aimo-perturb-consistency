#!/bin/sh
# Build ablation bundles: ./ablate.sh typo distract none -> dist/ablate-<name>.zip (Small track).
set -e
mkdir -p dist
for a in "$@"; do
  d=$(mktemp -d)
  sed "s/\"PC_PERTURBATIONS\", \"typo,distract,rename\"/\"PC_PERTURBATIONS\", \"$a\"/" solution.py > "$d/solution.py"
  grep -q "\"PC_PERTURBATIONS\", \"$a\"" "$d/solution.py"
  : > "$d/small.txt"
  (cd "$d" && zip -q "$OLDPWD/dist/ablate-$a.zip" solution.py small.txt)
done
