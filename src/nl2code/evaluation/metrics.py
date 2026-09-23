"""Exact match and syntax-only metrics; generated code is never executed."""

import ast
from collections import defaultdict

from nl2code.data.normalize import normalize_newlines


def _clean(code):
    return normalize_newlines(code).strip()


def _metric(count, total):
    return {"matched": count, "total": total, "value": count / total if total else 0.0}


def evaluate(rows, predictions):
    predicted = {row["record_id"]: row["prediction"] for row in predictions}
    references = defaultdict(set)
    intent_predictions = {}
    row_matches = 0
    for row in rows:
        intent = row["normalized_intent"]
        reference = _clean(row["snippet"])
        references[intent].add(reference)
        prediction = predicted[row["record_id"]]
        intent_predictions[intent] = prediction
        row_matches += _clean(prediction) == reference
    matches = sum(_clean(intent_predictions[intent]) in refs for intent, refs in references.items())
    valid = 0
    for prediction in intent_predictions.values():
        if not prediction.strip():
            continue
        try:
            ast.parse(prediction)
        except (SyntaxError, ValueError):
            continue
        valid += 1
    total = len(references)
    return {
        "intent_any_reference_exact_match": _metric(matches, total),
        "python_syntax_validity": {
            "valid": valid,
            "total": total,
            "value": valid / total if total else 0.0,
        },
        "row_exact_match": {**_metric(row_matches, len(rows)), "diagnostic": True},
        "total_records": len(rows),
        "unique_normalized_intent_groups": total,
        "multi_reference_group_count": sum(len(refs) > 1 for refs in references.values()),
        "max_references_per_group": max(map(len, references.values()), default=0),
        "empty_prediction_count": sum(not p.strip() for p in intent_predictions.values()),
    }
