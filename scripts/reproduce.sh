#!/bin/bash
# Reproduce the manifest from the BONES-SEED SOMA Uniform BVH.
#
# Usage: scripts/reproduce.sh [WORK_DIR]
# Needs a Hugging Face token with access to bones-studio/seed (accept its license first),
# about 60 GB of free disk, and the packages in requirements.txt. Uses every CPU.
# Ends by checking the outputs against SHA256SUMS.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mkdir -p "${1:-work}" && cd "${1:-work}" && pwd)"
REV=2f59b2077b9da34dd4e43618e705c7cb962c9a66
ARCHIVE_SHA=0c1fd91628d2644d0d7c0b735ae64faa5fa79834280bc1ce7860a871e55f912c

if [ ! -f "$WORK/soma_uniform.tar.gz" ]; then
  python -I -c "import sys; from huggingface_hub import hf_hub_download as d
d('bones-studio/seed', 'soma_uniform.tar.gz', repo_type='dataset', revision='$REV', local_dir=sys.argv[1])" "$WORK"
fi
echo "$ARCHIVE_SHA  $WORK/soma_uniform.tar.gz" | sha256sum -c -

if command -v rapidgzip >/dev/null; then UNZIP=(rapidgzip -d -c); else UNZIP=(gzip -d -c); fi
if [ ! -f "$WORK/features/meta.json" ]; then
  rm -rf "$WORK/features"
  "${UNZIP[@]}" "$WORK/soma_uniform.tar.gz" | python -I "$HERE/scripts/bvh_features.py" "$WORK/features"
fi
python -I "$HERE/scripts/detect.py" "$WORK/features" "$WORK/detections.json"
python -I "$HERE/scripts/make_manifest.py" "$WORK/detections.json" "$WORK/manifest"
(cd "$WORK/manifest" && sha256sum -c "$HERE/SHA256SUMS")
