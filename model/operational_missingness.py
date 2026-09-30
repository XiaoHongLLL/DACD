


from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}
HEALTHY_POST_STATES = {"", "stable_success", "success", "healthy", "ready", "complete"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def parse_bool(value):
    text = str(value or "").strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return None


def parse_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def index_run_dirs(roots: list[Path], state_filename: str) -> dict[str, Path]:
    pass
    indexed: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and (child / state_filename).exists():
                indexed.setdefault(child.name, child)
        for state_path in root.rglob(state_filename):
            indexed.setdefault(state_path.parent.name, state_path.parent)
    return indexed


def index_status_rows(paths: list[Path]) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in read_csv(path):
            run_id = str(row.get("run_id") or "").strip()
            if run_id:
                indexed[run_id] = row
    return indexed


def index_inventory_rows(paths: list[Path]) -> dict[str, list[dict[str, str]]]:
    indexed: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        for row in read_csv(path):
            run_id = str(row.get("run_id") or "").strip()
            if run_id:
                indexed[run_id].append(row)
    return dict(indexed)


def deployments_from_state(state: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    deployments = state.get("deployments", [])
    if not isinstance(deployments, list):
        return out
    for deployment in deployments:
        if not isinstance(deployment, dict):
            continue
        name = str(deployment.get("name") or "").strip()
        if name:
            out[name] = deployment
    return out


def deployment_healthy(deployment: dict, min_available: int) -> bool:
    replicas = parse_int(deployment.get("replicas"), 0)
    available = parse_int(deployment.get("available"), 0)
    ready = parse_bool(deployment.get("ready"))
    if replicas < min_available or available < min_available:
        return False
    if available < replicas:
        return False
    if ready is False:
        return False
    return True


def status_failure_reasons(row: dict[str, str]) -> list[str]:
    pass
    reasons: list[str] = []
    deployment_failed = parse_bool(row.get("deployment_failed"))
    deployment_success = parse_bool(row.get("deployment_success"))
    readiness_success = parse_bool(row.get("readiness_success"))
    if deployment_failed is True:
        reasons.append("status:deployment_failed")
    if deployment_success is False:
        reasons.append("status:deployment_unsuccessful")
    if readiness_success is False:
        reasons.append("status:readiness_failed")
    post_state = str(row.get("post_change_state") or "").strip().lower()
    if post_state not in HEALTHY_POST_STATES:
        reasons.append(f"status:post_change_state={post_state}")
    return reasons


def inventory_failure_reasons(
    rows: list[dict[str, str]],
    service_prefixes: tuple[str, ...],
    min_pre_events: int,
    max_post_events: int,
) -> list[str]:
    pass
    reasons: list[str] = []
    for row in rows:
        service = str(row.get("service") or "").strip()
        if not service or (service_prefixes and not service.startswith(service_prefixes)):
            continue
        if parse_bool(row.get("collector_complete")) is not True:
            continue
        pre_events = parse_int(row.get("pre_change_events"), 0)
        post_events = parse_int(row.get("post_change_events"), 0)
        if pre_events >= min_pre_events and post_events <= max_post_events:
            reasons.append(
                f"inventory:service_silence={service}:{pre_events}->{post_events}"
            )
    return reasons


def state_failure_reasons(
    state: dict,
    required_services: list[str],
    min_available: int,
) -> list[str]:
    reasons: list[str] = []
    deployments_available = parse_bool(state.get("deployments_available"))
    if deployments_available is False:
        reasons.append("state:deployments_unavailable")
    deployments = deployments_from_state(state)
    for service in required_services:
        deployment = deployments.get(service)
        if deployment is None:
            reasons.append(f"state:required_service_absent={service}")
            continue
        replicas = parse_int(deployment.get("replicas"), 0)
        available = parse_int(deployment.get("available"), 0)
        ready = parse_bool(deployment.get("ready"))
        if replicas < min_available:
            reasons.append(f"state:zero_desired_replicas={service}")
        elif available < min_available:
            reasons.append(f"state:zero_available_replicas={service}")
        elif available < replicas:
            reasons.append(f"state:partial_availability={service}:{available}/{replicas}")
        if ready is False:
            reasons.append(f"state:unready={service}")
    return reasons


def calibrate_required_services(
    reference_run_ids: list[str],
    run_dirs: dict[str, Path],
    state_filename: str,
    service_prefixes: tuple[str, ...],
    min_prevalence: float,
    min_healthy_prevalence: float,
    min_available: int,
) -> tuple[list[str], dict]:
    observed_references = 0
    present = Counter()
    healthy = Counter()
    for run_id in reference_run_ids:
        run_dir = run_dirs.get(run_id)
        if not run_dir:
            continue
        state = read_json(run_dir / state_filename)
        deployments = deployments_from_state(state)
        if not deployments:
            continue
        observed_references += 1
        for service, deployment in deployments.items():
            if service_prefixes and not service.startswith(service_prefixes):
                continue
            present[service] += 1
            if deployment_healthy(deployment, min_available):
                healthy[service] += 1

    required: list[str] = []
    details: dict[str, dict] = {}
    denominator = max(observed_references, 1)
    for service in sorted(present):
        present_rate = present[service] / denominator
        healthy_rate = healthy[service] / denominator
        selected = (
            present_rate >= min_prevalence
            and healthy_rate >= min_healthy_prevalence
        )
        details[service] = {
            "present_reference_runs": present[service],
            "healthy_reference_runs": healthy[service],
            "present_rate": present_rate,
            "healthy_rate": healthy_rate,
            "required": selected,
        }
        if selected:
            required.append(service)
    return required, {
        "reference_run_count": len(reference_run_ids),
        "reference_state_observed_count": observed_references,
        "required_services": required,
        "service_statistics": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-root", action="append", default=[])
    parser.add_argument("--status-csv", action="append", default=[])
    parser.add_argument("--service-inventory-csv", action="append", default=[])
    parser.add_argument("--split", default="test")
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--state-filename", default="cloud_state_after.json")
    parser.add_argument("--service-prefixes", default="ts-")
    parser.add_argument("--min-reference-prevalence", type=float, default=0.90)
    parser.add_argument("--min-reference-healthy-prevalence", type=float, default=0.90)
    parser.add_argument("--min-available", type=int, default=1)
    parser.add_argument("--inventory-min-pre-events", type=int, default=10)
    parser.add_argument("--inventory-max-post-events", type=int, default=0)
    parser.add_argument("--require-complete-evidence", action="store_true")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    if not args.run_root and not args.status_csv and not args.service_inventory_csv:
        parser.error(
            "provide at least one --run-root, --status-csv, or --service-inventory-csv"
        )

    data_dir = Path(args.data_dir)
    annotations = read_csv(data_dir / "annotation.csv")


    identities = [
        {"run_id": str(row.get("run_id") or "").strip(), "split": str(row.get("split") or "").strip()}
        for row in annotations
        if str(row.get("run_id") or "").strip()
    ]
    reference_ids = [row["run_id"] for row in identities if row["split"] == args.calibration_split]
    evaluation_ids = [row["run_id"] for row in identities if row["split"] == args.split]

    run_dirs = index_run_dirs([Path(item) for item in args.run_root], args.state_filename)
    status_rows = index_status_rows([Path(item) for item in args.status_csv])
    inventory_rows = index_inventory_rows(
        [Path(item) for item in args.service_inventory_csv]
    )
    prefixes = tuple(item.strip() for item in args.service_prefixes.split(",") if item.strip())
    required_services, calibration = calibrate_required_services(
        reference_ids,
        run_dirs,
        args.state_filename,
        prefixes,
        args.min_reference_prevalence,
        args.min_reference_healthy_prevalence,
        args.min_available,
    )

    rows: list[dict] = []
    missing_evidence_runs: list[str] = []
    for run_id in evaluation_ids:
        run_dir = run_dirs.get(run_id)
        state = read_json(run_dir / args.state_filename) if run_dir else {}
        status = status_rows.get(run_id, {})
        inventory = inventory_rows.get(run_id, [])
        has_state = bool(deployments_from_state(state)) or "deployments_available" in state
        has_status = any(
            key in status
            for key in ("deployment_failed", "deployment_success", "readiness_success", "post_change_state")
        )
        inventory_complete = bool(inventory) and all(
            parse_bool(row.get("collector_complete")) is True for row in inventory
        )
        reasons: list[str] = []
        if has_state:
            reasons.extend(state_failure_reasons(state, required_services, args.min_available))
        if has_status:
            reasons.extend(status_failure_reasons(status))
        if inventory_complete:
            reasons.extend(
                inventory_failure_reasons(
                    inventory,
                    prefixes,
                    args.inventory_min_pre_events,
                    args.inventory_max_post_events,
                )
            )
        reasons = sorted(set(reasons))
        observed = has_state or has_status or inventory_complete
        if not observed:
            missing_evidence_runs.append(run_id)
        rows.append({
            "run_id": run_id,
            "evidence_observed": int(observed),
            "state_observed": int(has_state),
            "status_observed": int(has_status),
            "inventory_observed": int(inventory_complete),
            "missing_alarm": int(bool(reasons)),
            "missing_reason_count": len(reasons),
            "missing_reasons": "|".join(reasons),
        })

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_name(out_prefix.name + "_evidence.csv")
    json_path = out_prefix.with_name(out_prefix.name + "_calibration.json")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["run_id"])
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "protocol": "label-free-operational-missingness-v1",
        "data_dir": str(data_dir),
        "split": args.split,
        "calibration_split": args.calibration_split,
        "forbidden_inputs_not_consumed": [
            "semantic_label",
            "benchmark_label",
            "change_target_component_id",
            "affected_component_ids",
            "oracle_component_ids",
        ],
        "decision_sources": [
            "cloud_state_after.json",
            "operational status CSV fields",
            "label-free service inventory pre/post event counts",
        ],
        "calibration": calibration,
        "evaluation_run_count": len(evaluation_ids),
        "evidence_observed_count": len(evaluation_ids) - len(missing_evidence_runs),
        "evidence_coverage": (
            (len(evaluation_ids) - len(missing_evidence_runs)) / len(evaluation_ids)
            if evaluation_ids else 0.0
        ),
        "missing_alarm_count": sum(int(row["missing_alarm"]) for row in rows),
        "missing_evidence_run_ids": missing_evidence_runs,
        "parameters": {
            "service_prefixes": list(prefixes),
            "min_reference_prevalence": args.min_reference_prevalence,
            "min_reference_healthy_prevalence": args.min_reference_healthy_prevalence,
            "min_available": args.min_available,
            "inventory_min_pre_events": args.inventory_min_pre_events,
            "inventory_max_post_events": args.inventory_max_post_events,
        },
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {csv_path}")
    print(f"[OK] Wrote {json_path}")
    print(json.dumps({
        "evaluation_runs": len(evaluation_ids),
        "evidence_observed": payload["evidence_observed_count"],
        "evidence_coverage": payload["evidence_coverage"],
        "missing_alarms": payload["missing_alarm_count"],
        "required_services": required_services,
    }, ensure_ascii=False, indent=2))

    if args.require_complete_evidence and missing_evidence_runs:
        print(
            "[ERROR] Operational evidence is missing for "
            f"{len(missing_evidence_runs)} run(s); refusing a paper-facing result."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
