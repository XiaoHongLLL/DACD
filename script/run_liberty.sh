#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode=${1:-all}
case "$mode" in train|eval|all) ;; *) echo "Usage: bash $0 [train|eval|all]" >&2; exit 2;; esac
python_bin=${PYTHON_BIN:-python}
export CUDA_VISIBLE_DEVICES=${DEVICE:-0}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
data_dir=${DATA_DIR:-data/labeled_liberty}
checkpoint=${CHECKPOINT:-checkpoints/liberty.pth}
result_dir=${RESULT_DIR:-results/liberty}
workers=${NUM_WORKERS:-8}
mkdir -p "$(dirname "$checkpoint")" "$result_dir"
for split in train dev test; do
  [[ -f "$data_dir/$split.pkl" ]] || { echo "Missing $data_dir/$split.pkl" >&2; exit 1; }
done
common=(
  -data "$data_dir/" -normalize log -d_model 128 -d_inner_hid 256
  -n_head 4 -n_layers 4 -d_k 32 -d_v 32 -dropout 0.1 -type_head gmm
  -lr 5e-5 -loss_weighting adaptive -fm_loss_weight 1.0 -loss_lambda 1.0
  -fm_sigma 0.01 -solver_method euler -clamp_threshold 6.0 -flow_cond_clip 5.0
  -n_samples 100 -checkpoint_metric valid_acc -checkpoint_min_delta 0.0001
  -seed 2023 -num_workers "$workers" -time_norm_guard_threshold 20.0
)
if [[ "$mode" == train || "$mode" == all ]]; then
  [[ ! -e "$checkpoint" ]] || { echo "Checkpoint exists; use eval or a new CHECKPOINT path." >&2; exit 1; }
  "$python_bin" model/public/main.py "${common[@]}" \
    -batch_size 64 -epoch 60 -eval_epoch 1000000 \
    -solver_step_size 0.05 -save_path "$checkpoint" 2>&1 | tee "$result_dir/train.log"
fi
if [[ "$mode" == eval || "$mode" == all ]]; then
  [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint; run train first." >&2; exit 1; }
  "$python_bin" model/evaluate_public.py public "${common[@]}" \
    -batch_size 256 -just_eval -eval_reliability -load_path_name "$checkpoint" \
    -solver_step_size 0.01 -uncertainty_mc 8 -calibration_max_size 200000 \
    -anomaly_quantile 0.99 -uncertainty_quantile 0.95 \
    -anomaly_score_mode zscore -type_score_weight 1.0 -time_score_weight 0.5 \
    -segment_score_mode alert_fraction -segment_topk 3 -type_entropy_weight 0 \
    -use_ensemble_correction -use_drift_adapter \
    -ensemble_k 20 -ensemble_samples 16 -ensemble_kernel 0.2 \
    -ensemble_noise_scale 0.1 -ensemble_correction_weight 1.0 \
    -ensemble_support_quantile 0.95 -ensemble_max_reference 4096 -ensemble_max_search 50000 \
    -save_result "$result_dir/evaluation" 2>&1 | tee "$result_dir/eval.log"
  echo "Metrics: $result_dir/metrics.csv"
fi
