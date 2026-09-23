"""Transitive leakage groups and deterministic group-level hash assignment."""

from collections import defaultdict

from nl2code.data.contract import sha256, stable_json

SPLIT_POLICY = "connected-components-hash-v1"
RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
SPLITS = tuple(RATIOS)


def assign_splits(records: list[dict[str, str]], seed: int) -> dict[str, int]:
    """Set group_id/split on each record, then prove both keys are disjoint."""
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for key in ("question_id", "normalized_intent"):
            identity = (key, record[key])
            if identity in seen:
                parent[find(index)] = find(seen[identity])
            else:
                seen[identity] = index

    components: dict[int, list[dict[str, str]]] = defaultdict(list)
    for index, record in enumerate(records):
        components[find(index)].append(record)

    group_counts = dict.fromkeys(SPLITS, 0)
    for component in components.values():
        keys = {
            key: sorted({record[key] for record in component})
            for key in ("question_id", "normalized_intent")
        }
        group_id = sha256(stable_json(keys))
        bucket = int(sha256(stable_json([SPLIT_POLICY, seed, group_id])), 16) % 10000
        split = "train" if bucket < 8000 else "validation" if bucket < 9000 else "test"
        group_counts[split] += 1
        for record in component:
            record["group_id"] = group_id
            record["split"] = split

    for key in ("question_id", "normalized_intent"):
        owners: dict[str, str] = {}
        for record in records:
            owner = owners.setdefault(record[key], record["split"])
            if owner != record["split"]:
                raise AssertionError(f"cross-split {key}: {record[key]}")
    return group_counts
