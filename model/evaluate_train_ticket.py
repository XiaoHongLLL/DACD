


from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path


INCLUDED_BENCHMARKS = {
    "expected_drift",
    "successful_no_drift",
    "unexpected_drift",
    "unexpected_without_observable_log_drift",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, score


def load_split(data_dir: Path, split: str):
    with (data_dir / f"{split}.pkl").open("rb") as handle:
        return pickle.load(handle, encoding="latin-1")[split]


def infer_batch_size(event_rows: list[dict[str, str]]) -> int:
    if not event_rows:
        raise ValueError("events CSV is empty")
    return max(safe_int(row.get("sequence_index"), -1) for row in event_rows) + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--events-csv", required=True)
    parser.add_argument("--missing-evidence-csv", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=0, help="0 infers it from sequence_index.")
    parser.add_argument("--min-unexpected-events", type=int, default=1)
    parser.add_argument("--min-unexpected-fraction", type=float, default=0.001)
    parser.add_argument("--require-complete-evidence", action="store_true")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data = load_split(data_dir, args.split)
    annotations = {row["run_id"]: row for row in read_csv(data_dir / "annotation.csv")}
    event_rows = read_csv(Path(args.events_csv))
    batch_size = args.batch_size or infer_batch_size(event_rows)
    evidence_rows = read_csv(Path(args.missing_evidence_csv))
    forbidden_evidence_columns = {
        "semantic_label", "benchmark_label", "true_id", "true_label",
        "change_target_component_id", "affected_component_ids", "oracle_component_ids",
    }
    leaked_columns = forbidden_evidence_columns.intersection(evidence_rows[0] if evidence_rows else {})
    if leaked_columns:
        raise ValueError(f"missing-evidence CSV contains forbidden label/target columns: {sorted(leaked_columns)}")
    missing_evidence = {row.get("run_id", ""): row for row in evidence_rows}

    by_run = defaultdict(lambda: Counter(normal=0, expected=0, unexpected=0))
    skipped_rows = 0
    for row in event_rows:
        batch_index = safe_int(row.get("batch_index"), -1)
        sequence_index = safe_int(row.get("sequence_index"), -1)
        global_index = batch_index * batch_size + sequence_index
        if global_index < 0 or global_index >= len(data) or not data[global_index]:
            skipped_rows += 1
            continue
        run_id = str(data[global_index][0].get("run_id") or "")
        if not run_id:
            skipped_rows += 1
            continue
        anomaly = safe_float(row.get("anomaly_score"), float("-inf"))
        gamma = safe_float(row.get("gamma_anomaly"), float("inf"))
        uncertainty = safe_float(row.get("uncertainty_score"), float("-inf"))
        delta = safe_float(row.get("delta_uncertainty"), float("inf"))
        if anomaly <= gamma:
            by_run[run_id]["normal"] += 1
        elif uncertainty <= delta:
            by_run[run_id]["expected"] += 1
        else:
            by_run[run_id]["unexpected"] += 1

    expected_run_ids = {
        row["run_id"]
        for row in annotations.values()
        if row.get("split") == args.split and row.get("benchmark_label") in INCLUDED_BENCHMARKS
    }
    absent_from_events = sorted(expected_run_ids.difference(by_run))
    if absent_from_events:
        raise ValueError(f"no event rows mapped to {len(absent_from_events)} evaluation run(s): {absent_from_events[:5]}")

    evidence_not_observed: list[str] = []
    rows: list[dict] = []
    confusion = Counter()
    subtype_confusion = Counter()
    for run_id in sorted(expected_run_ids):
        annotation = annotations[run_id]
        counts = by_run[run_id]
        total = sum(counts.values())
        unexpected_fraction = counts["unexpected"] / total if total else 0.0
        base_alarm = (
            counts["unexpected"] >= args.min_unexpected_events
            and unexpected_fraction >= args.min_unexpected_fraction
        )
        evidence = missing_evidence.get(run_id, {})
        evidence_observed = safe_int(evidence.get("evidence_observed"), 0) == 1
        if not evidence_observed:
            evidence_not_observed.append(run_id)
        missing_alarm = evidence_observed and safe_int(evidence.get("missing_alarm"), 0) == 1
        final_alarm = base_alarm or missing_alarm
        true_label = "Expected" if annotation.get("semantic_label") == "expected" else "Unexpected"
        pred_label = "Unexpected" if final_alarm else "Expected"
        confusion[(true_label, pred_label)] += 1
        subtype_confusion[(annotation.get("benchmark_label", ""), pred_label)] += 1
        rows.append({
            "run_id": run_id,
            "semantic_label": annotation.get("semantic_label", ""),
            "benchmark_label": annotation.get("benchmark_label", ""),
            "true_label": true_label,
            "pred_label": pred_label,
            "decision_reason": "operational_missingness" if missing_alarm else ("a_and_u" if base_alarm else "accepted"),
            "base_au_alarm": int(base_alarm),
            "missing_alarm": int(missing_alarm),
            "evidence_observed": int(evidence_observed),
            "missing_reasons": evidence.get("missing_reasons", ""),
            "events_seen": total,
            "normal_events": counts["normal"],
            "expected_events": counts["expected"],
            "unexpected_events": counts["unexpected"],
            "unexpected_fraction": unexpected_fraction,
        })

    metrics = {}
    for label in ("Expected", "Unexpected"):
        tp = confusion[(label, label)]
        fp = sum(count for (true, pred), count in confusion.items() if true != label and pred == label)
        fn = sum(count for (true, pred), count in confusion.items() if true == label and pred != label)
        precision, recall, score = f1(tp, fp, fn)
        metrics[f"{label}_Precision"] = precision
        metrics[f"{label}_Recall"] = recall
        metrics[f"{label}_F1"] = score
    metrics["Macro_F1"] = (metrics["Expected_F1"] + metrics["Unexpected_F1"]) / 2
    metrics["Accuracy"] = sum(confusion[(x, x)] for x in ("Expected", "Unexpected")) / max(len(rows), 1)
    metrics["Unexpected_False_Acceptance_Rate"] = (
        confusion[("Unexpected", "Expected")]
        / max(sum(count for (true, _), count in confusion.items() if true == "Unexpected"), 1)
    )
    metrics["Expected_False_Alarm_Rate"] = (
        confusion[("Expected", "Unexpected")]
        / max(sum(count for (true, _), count in confusion.items() if true == "Expected"), 1)
    )

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = out_prefix.with_name(out_prefix.name + "_predictions.csv")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.json")
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["run_id"])
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "method": "DACD-unsupervised-a-u-m",
        "decision_rule": {
            "event": "a<=gamma:Normal; a>gamma,u<=delta:Expected; a>gamma,u>delta:Unexpected",
            "run": "Unexpected iff m=1 or the fixed a/u event-count rule fires; otherwise Accepted Expected",
            "missingness_precedence": "m=1 is a hard operational veto",
            "min_unexpected_events": args.min_unexpected_events,
            "min_unexpected_fraction": args.min_unexpected_fraction,
        },
        "supervision": {
            "eu_labels_used_for_training": False,
            "eu_labels_used_for_threshold_selection": False,
            "dev_threshold_search": False,
            "labels_used_after_predictions_for_metrics_only": True,
        },
        "batch_size": batch_size,
        "event_rows": len(event_rows),
        "skipped_event_rows": skipped_rows,
        "run_count": len(rows),
        "evidence_observed_count": len(rows) - len(evidence_not_observed),
        "evidence_coverage": (len(rows) - len(evidence_not_observed)) / max(len(rows), 1),
        "evidence_not_observed_run_ids": evidence_not_observed,
        "alarm_counts": {
            "base_au": sum(row["base_au_alarm"] for row in rows),
            "missingness": sum(row["missing_alarm"] for row in rows),
            "union": sum(row["pred_label"] == "Unexpected" for row in rows),
        },
        "confusion": {f"{true}->{pred}": count for (true, pred), count in sorted(confusion.items())},
        "subtype_confusion": [
            {"benchmark_label": subtype, "pred_label": pred, "count": count}
            for (subtype, pred), count in sorted(subtype_confusion.items())
        ],
        "metrics": metrics,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote {predictions_path}")
    print(f"[OK] Wrote {summary_path}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.require_complete_evidence and evidence_not_observed:
        print(
            "[ERROR] Operational evidence is incomplete for "
            f"{len(evidence_not_observed)} run(s); metrics above are diagnostic only."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
