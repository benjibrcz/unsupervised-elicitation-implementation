#!/usr/bin/env bash
set -euo pipefail

# One-shot runner for the Coherent Myths analysis (baseline, myth-seeded, compare)
# Usage: bash scripts/run_myths_all.sh

BASE_MODEL=${BASE_MODEL:-meta-llama/Meta-Llama-3.1-405B-Instruct}
ICM_STEPS=${ICM_STEPS:-300}
ICM_TARGET=${ICM_TARGET:-128}
ALPHA=${ALPHA:-40}
CONTEXT_CAP=${CONTEXT_CAP:-96}
EVAL_K=${EVAL_K:-64}
EVAL_MODE=${EVAL_MODE:-strict_text}
TRAIN=${TRAIN:-data/coherent_myths_train.json}
TEST=${TEST:-data/coherent_myths_test.json}

# echo "Running Myths baseline..."
# python -m src.run \
#   --train "$TRAIN" \
#   --test  "$TEST" \
#   --base_model "$BASE_MODEL" \
#   --icm_steps "$ICM_STEPS" --icm_target_labels "$ICM_TARGET" --alpha "$ALPHA" \
#   --context_cap "$CONTEXT_CAP" --eval_k "$EVAL_K" --eval_mode "$EVAL_MODE" \
#   --results results_myths_large.json --figure results_myths_large.png

echo "Running Myths myth-seeded ICM..."
python -m src.run \
  --train "$TRAIN" \
  --test  "$TEST" \
  --base_model "$BASE_MODEL" \
  --icm_steps "$ICM_STEPS" --icm_target_labels "$ICM_TARGET" --alpha "$ALPHA" \
  --context_cap "$CONTEXT_CAP" --eval_k "$EVAL_K" --eval_mode "$EVAL_MODE" \
  --icm_seed_myth \
  --results results_myths_large_mythseed.json --figure results_myths_large_mythseed.png

echo "Running Neutral vs Myth seed comparison..."
python -m src.run_myths_compare \
  --train "$TRAIN" \
  --test  "$TEST" \
  --base_model "$BASE_MODEL" \
  --icm_steps "$ICM_STEPS" --icm_target_labels "$ICM_TARGET" --alpha "$ALPHA" \
  --context_cap "$CONTEXT_CAP" --eval_k "$EVAL_K" --eval_mode "$EVAL_MODE" \
  --out_prefix results_myths_large_compare

echo "Done. Artifacts: results_myths_large*.{json,png} and results_myths_large_compare.{json,png}"


