#!/usr/bin/env bash
# Usage: bash setup.sh            (CPU)
#        FORGE_CUDA=1 bash setup.sh   (NVIDIA GPU)
#        FORGE_SHAPE=1 bash setup.sh  (also install Shap-E backend)
set -euo pipefail
cd "$(dirname "$0")"

python -m pip install --upgrade pip
if [ "${FORGE_CUDA:-0}" = "1" ]; then
  pip install torch torchvision
else
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
pip install -r requirements.txt

mkdir -p third_party
[ -d third_party/TripoSR ] || git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git third_party/TripoSR

# TripoSR pins old versions that clash with current diffusers; install them unpinned.
grep -v -E '^(gradio|git\+|#|$)' third_party/TripoSR/requirements.txt | sed -E 's/[=<>~!].*//' > /tmp/triposr-req.txt
pip install -r /tmp/triposr-req.txt
pip install git+https://github.com/tatsy/torchmcubes.git \
  || pip install --no-build-isolation git+https://github.com/tatsy/torchmcubes.git

if [ "${FORGE_SHAPE:-0}" = "1" ]; then
  pip install git+https://github.com/openai/shap-e.git
fi
echo "Setup complete."
