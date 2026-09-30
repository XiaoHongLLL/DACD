#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode=${1:-all}
case "$mode" in train|eval|all) ;; *) echo "Usage: bash $0 [train|eval|all]" >&2; exit 2;; esac
python_bin=${PYTHON_BIN:-python}
export CUDA_VISIBLE_DEVICES=${DEVICE:-0}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
data_dir=${DATA_DIR:-data/train_ticket}
checkpoint=${CHECKPOINT:-checkpoints/train_ticket.pth}
result_dir=${RESULT_DIR:-results/train_ticket}
workers=${NUM_WORKERS:-8}
mkdir -p "$(dirname "$checkpoint")" "$result_dir"
for name in train.pkl dev.pkl test.pkl annotation.csv service_inventory_new.csv; do
  [[ -f "$data_dir/$name" ]] || { echo "Missing $data_dir/$name" >&2; exit 1; }
done
common=(
  -data "$data_dir/" -normalize log -d_model 128 -d_inner_hid 256
  -n_head 4 -n_layers 4 -d_k 32 -d_v 32 -dropout 0.1 -type_head gmm -lr 5e-5
  -loss_weighting fixed -fm_loss_weight 0.05 -loss_lambda 5.0 -fm_sigma 0.5
  -solver_method euler -clamp_threshold 6.0 -flow_cond_clip 5.0 -min_max_len 0 -n_samples 100
  -checkpoint_metric valid_acc -checkpoint_min_delta 0.0001
  -context_mask_prob 0.1 -context_mask_min_history 1 -seed 2023 -num_workers "$workers"
)
if [[ "$mode" == train || "$mode" == all ]]; then
  [[ ! -e "$checkpoint" ]] || { echo "Checkpoint exists; use eval or a new CHECKPOINT path." >&2; exit 1; }
  "$python_bin" model/current/main.py "${common[@]}" \
    -batch_size 64 -epoch 60 -eval_epoch 1000000 -solver_step_size 0.05 \
    -save_path "$checkpoint" 2>&1 | tee "$result_dir/train.log"
fi
if [[ "$mode" == eval || "$mode" == all ]]; then
  [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint; run train first." >&2; exit 1; }
  "$python_bin" model/current/main.py "${common[@]}" \
    -batch_size 128 -just_eval -eval_reliability -reliability_exact_nll \
    -load_path_name "$checkpoint" -solver_step_size 0.01 \
    -rq4_candidate_mode score -anomaly_score_mode zscore -type_score_weight 1.0 -time_score_weight 0.5 \
    -uncertainty_mc 4 -flow_mismatch_mode robust -flow_mismatch_time_steps 2 \
    -flow_mismatch_path_aggregation q10 -flow_mismatch_time_weight t2 \
    -calibration_split train -decision_policy score_then_uncertainty \
    -anomaly_quantile 0.99 -uncertainty_quantile 0.95 -calibration_max_size 200000 \
    -save_rq4_event_details -rq4_event_detail_include_normal -rq4_event_detail_max 1000000 \
    -save_result "$result_dir/test" 2>&1 | tee "$result_dir/eval.log"
  "$python_bin" model/operational_missingness.py \
    --data-dir "$data_dir" --split test --calibration-split train \
    --service-inventory-csv "$data_dir/service_inventory_new.csv" \
    --inventory-min-pre-events 10 --inventory-max-post-events 0 --service-prefixes ts- \
    --require-complete-evidence --output-prefix "$result_dir/missingness"
  "$python_bin" model/evaluate_train_ticket.py \
    --data-dir "$data_dir" --split test --events-csv "$result_dir/test_rq4_events.csv" \
    --missing-evidence-csv "$result_dir/missingness_evidence.csv" --batch-size 128 \
    --min-unexpected-events 1 --min-unexpected-fraction 0.001 \
    --require-complete-evidence --output-prefix "$result_dir/metrics"
fi
