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

Train-Ticket is v0.6, 368 runs: train30/adapt40/dev40/test120/archive138.
All five splits are included; only test120 is the formal evaluation set.
The model inputs retain 596,793 window positions, including 12,473 first positions
needed as context. Excluding these yields the 584,320 scored positions above;
these are window occurrences, not necessarily unique raw messages. No prediction files are bundled.

Public evaluation preserves the screenshot's historical adaptive profiles, including
targeted HDFS; `model/public` freezes the older public engine, while `model/current`
supplies targeted HDFS and Train-Ticket. These are required implementations, not backups.
Public `metrics.csv` reports both historical target-only labels and full-sequence
labels; only target-only corresponds to the screenshot. Profiles/refinement contribute
to the historical results, which are not a pure neural a/u ablation.
Original benchmark checkpoints are not bundled; fresh training does not guarantee
identical rounded screenshot numbers. Train-Ticket fresh training selects valid_acc
on the included v0.6 dev split, not the historical v0.4 dev split; its evaluation
uses the fixed a/u/m rule on test120. Exact historical evaluation needs the original
frozen checkpoint via `CHECKPOINT=... bash run_train_ticket.sh eval`.

For GitHub, extract this archive and push the directory with Git. Individual files
are below 100 MiB; some exceed the web uploader's 25 MiB limit.
