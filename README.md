# DACD

Model source, six train/evaluate entry points, and the exact preprocessed dataset splits.
Linux + Bash; Python 3.9. Local smoke checks use PyTorch 1.9.0+cu111,
NumPy 1.26.4 and torchdiffeq 0.2.5 (full benchmark training was not rerun).
Install a suitable PyTorch build, then `pip install numpy==1.26.4 torchdiffeq==0.2.5 tqdm pandas`.

From this directory (a CUDA GPU is recommended):

```bash
bash run_hdfs.sh
bash run_bgl.sh
bash run_thunderbird.sh
bash run_spirit.sh
bash run_liberty.sh
bash run_train_ticket.sh
```

Each entry defaults to training then evaluation; append `train` or `eval` for one stage.
Parameters are embedded. `DEVICE`, `PYTHON_BIN`, `NUM_WORKERS`, `DATA_DIR`,
`CHECKPOINT`, and `RESULT_DIR` may be set in the environment.
Checkpoints and generated outputs go to `checkpoints/` and `results/`.
Existing checkpoints are never overwritten by a training entry.

| Dataset | Event occurrences | Normal / Expected | Anomaly / Unexpected |
|---|---:|---:|---:|
| HDFS | 2,300,683 | 2,012,433 | 288,250 |
| BGL | 4,713,450 | 4,364,998 | 348,452 |
| Thunderbird | 2,000,000 | 1,993,381 | 6,619 |
| Spirit | 2,000,000 | 1,538,203 | 461,797 |
| Liberty | 2,000,000 | 1,169,860 | 830,140 |
| Train-Ticket (scored positions) | 584,320 | 158,746 | 425,574 |


