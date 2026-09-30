import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import main as engine
from preprocess.Dataset import EventData

original = engine.eval_reliability

def metrics(truth, pred):
    tp = sum(y == 1 and p == 1 for y, p in zip(truth, pred))
    tn = sum(y == 0 and p == 0 for y, p in zip(truth, pred))
    fp = sum(y == 0 and p == 1 for y, p in zip(truth, pred))
    fn = sum(y == 1 and p == 0 for y, p in zip(truth, pred))
    div = lambda n, d: n / d if d else 0.0
    return dict(TP=tp, TN=tn, FP=fp, FN=fn, Precision=div(tp,tp+fp),
                Recall=div(tp,tp+fn), F1=div(2*tp,2*tp+fp+fn),
                FPR=div(fp,fp+tn), FNR=div(fn,fn+tp))

def save(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def evaluate(model, calibration, test, opt):
    if not isinstance(test.sampler, torch.utils.data.SequentialSampler):
        raise ValueError('Evaluation requires sequential loading')
    for seq in calibration.dataset.raw_data:
        if any(EventData._event_to_binary_label(e) != 0 for e in seq):
            raise ValueError('Public calibration must contain only normal events')
    checkpoint = torch.load(opt.load_path_name, map_location='cpu')
    model.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
    captured = []
    update = engine.update_binary_counts
    def capture(counts, prefix, pred_positive, true_label, valid_mask):
        if prefix == 'OursOODSegment':
            if pred_positive.ndim != 1 or not bool(valid_mask.all()):
                raise ValueError('Incomplete sequence predictions')
            captured.extend(zip(pred_positive.long().cpu().tolist(), true_label.long().cpu().tolist()))
        return update(counts, prefix, pred_positive, true_label, valid_mask)
    engine.update_binary_counts = capture
    try:
        result = original(model, calibration, test, opt)
    finally:
        engine.update_binary_counts = update
    if len(captured) != len(test.dataset.raw_data) or not captured:
        raise ValueError('Prediction coverage mismatch')
    rows = []
    for i, (seq, (pred, target_truth)) in enumerate(zip(test.dataset.raw_data, captured)):
        labels = [EventData._event_to_binary_label(e) for e in seq]
        if any(y not in (0, 1) for y in labels) or int(any(labels[1:])) != target_truth:
            raise ValueError('Unknown labels or changed prediction order')
        rows.append(dict(sequence_id=i, prediction=pred, target_only_label=target_truth,
                         full_sequence_label=int(any(labels))))
    out = Path(opt.save_result).parent
    save(out / 'predictions.csv', rows)
    summary = [dict(LabelScope=scope, **metrics([r[scope] for r in rows],
               [r['prediction'] for r in rows])) for scope in ('target_only_label', 'full_sequence_label')]
    for key in ('TP', 'TN', 'FP', 'FN'):
        if summary[0][key] != result['OursOODSegment_' + key]:
            raise ValueError('Engine/summary mismatch')
    save(out / 'metrics.csv', summary)
    print(summary)
    return result

if __name__ == '__main__':
    engine.eval_reliability = evaluate
    engine.main()
