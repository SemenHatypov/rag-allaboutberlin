#!/usr/bin/env bash
# Mirror the current working tree to the Hugging Face Space.
#
# HF's pre-receive hook rejects binaries committed to plain git (it wants
# LFS/Xet). Instead of migrating the repo to LFS, we ship a single ORPHAN
# commit — no ancestors, so none of the older binary blobs are in the pushed
# range — with output/json/embeddings.npz stripped out. The Docker build
# regenerates that file from the corpus (see Dockerfile), so the Space still
# starts fast. GitHub main and Streamlit Cloud keep the committed .npz and are
# untouched by this.
#
# Env: HF_TOKEN (write-scoped), HF_SPACE ("username/space-name").
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ] || [ -z "${HF_SPACE:-}" ]; then
  echo "HF_TOKEN secret or HF_SPACE variable not set; skipping deploy."
  exit 0
fi

sha="${GITHUB_SHA:-$(git rev-parse --short HEAD)}"
remote="https://${HF_SPACE%%/*}:${HF_TOKEN}@huggingface.co/spaces/${HF_SPACE}"

git config user.email "github-actions[bot]@users.noreply.github.com"
git config user.name "github-actions[bot]"

# Assemble the snapshot on a throwaway orphan branch.
git branch -D _hf_deploy 2>/dev/null || true
git checkout --orphan _hf_deploy
git rm --cached --quiet -f output/json/embeddings.npz 2>/dev/null || true
rm -f output/json/embeddings.npz
git add -A
git commit --quiet -m "Deploy snapshot from ${sha}"

git push --force "$remote" _hf_deploy:main
echo "Pushed snapshot to https://huggingface.co/spaces/${HF_SPACE}"
