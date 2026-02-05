#!/bin/bash
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate base # This might be redundant if the next activate works
conda activate ${PY}

python src/train.py train=false
# python src/train.py experiment=c train=false
# python src/train.py experiment=c trainer.max_epochs=1