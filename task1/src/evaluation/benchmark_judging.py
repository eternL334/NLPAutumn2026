"""Blinded packets, strict rating validation and preserved adjudication inputs."""

import json
from collections import Counter

import numpy as np

from src.evaluation.benchmark import ROOT, RUBRIC, read_jsonl, verify_frozen
from src.training.data import read_json, write_json

FIELDS = {"continuation": ["language", "readability", "coherence", "relevance", "predominant_degeneration", "uncertainty"],
          "qa": ["qa_grade"]}


def acceptable(row):
    if "qa_grade" in row:
        return row["qa_grade"] == "correct"
    return (row["language"] == "bulgarian" and all(row[k] >= 1 for k in ["readability", "coherence", "relevance"])
            and not row["predominant_degeneration"] and not row["uncertainty"])


def controls():
    return [
        {"task": "continuation", "prompt": "В неделя Мария реши да отиде на пазара. Там тя срещна своята съседка.",
         "response": "Двете се поздравиха и тръгнаха заедно към сергиите със зеленчуци. Мария избра домати и краставици за салатата, а съседката ѝ купи картофи за вечеря. После седнаха на пейката пред пазара и поговориха за децата си. Когато слънцето стана по-силно, всяка взе своята чанта и се прибра у дома.", "expected": True},
        {"task": "continuation", "prompt": "Преди да засадим цветята, трябва да подготвим саксиите и почвата.",
         "response": "На дъното на всяка саксия поставяме малко камъчета, за да може излишната вода да се оттича. След това добавяме почвата и правим малка дупка за корените. Поставяме растението внимателно, покриваме корените и притискаме леко пръстта. Накрая поливаме цветето и го оставяме на подходящо светло място.", "expected": True},
        {"task": "continuation", "prompt": "Мария излезе от дома си и тръгна към парка.",
         "response": "She walked along the river and watched the birds. When she reached the park, she sat on a bench and read her book. Later she met a friend and they returned home together.", "expected": False},
        {"task": "continuation", "prompt": "Реката преминава през града и се влива в морето.",
         "response": "Водата е вода. " * 20, "expected": False},
        {"task": "qa", "context": "Петър пристигна в Пловдив във вторник.", "question": "В кой ден пристигна Петър?",
         "reference": "Във вторник.", "response": "Във вторник.", "expected": True},
        {"task": "qa", "context": "Петър пристигна в Пловдив във вторник.", "question": "В кой ден пристигна Петър?",
         "reference": "Във вторник.", "response": "В петък.", "expected": False},
    ]


def packets():
    verify_frozen()
    if (ROOT / "judging/assignment.json").exists():
        raise FileExistsError("Judge packets already frozen; do not reshuffle after evaluation")
    items = {r["id"]: r for r in read_jsonl(ROOT / "items.jsonl")}
    generated = []
    for model in ["monolm", "qwen"]:
        rows = read_jsonl(ROOT / "generations" / f"{model}.jsonl")
        assert len(rows) == 100 and {r["id"] for r in rows} == set(items)
        assert all(r["model"] == model for r in rows)
        generated.extend(rows)
    rng = np.random.default_rng(42)
    double = set()
    for model in ["monolm", "qwen"]:
        for task, count in [("continuation", 12), ("qa", 8)]:
            candidates = [i for i, r in enumerate(generated) if r["model"] == model and r["task"] == task]
            double.update(int(i) for i in rng.choice(candidates, count, replace=False))
    assignment, payloads = {}, {1: [], 2: [], 3: []}
    for order, index in enumerate(rng.permutation(len(generated))):
        index = int(index)
        row = generated[index]
        item = items[row["id"]]
        opaque = rng.bytes(8).hex()
        primary = order % 3 + 1
        secondary = int(rng.choice([j for j in [1, 2, 3] if j != primary])) if index in double else None
        payload = {"id": opaque, "task": item["task"], "response": row["response"]}
        payload.update({k: item[k] for k in (["prompt"] if item["task"] == "continuation" else ["context", "question", "reference"])})
        assignment[opaque] = {"model": row["model"], "item_id": row["id"], "family": row["family"],
                              "task": row["task"], "primary": primary, "secondary": secondary, "control": False}
        payloads[primary].append(payload)
        if secondary:
            payloads[secondary].append(payload)
    for judge in [1, 2, 3]:
        for control in controls():
            opaque = rng.bytes(8).hex()
            payload = {"id": opaque, **{k: v for k, v in control.items() if k != "expected"}}
            assignment[opaque] = {"task": control["task"], "primary": judge, "secondary": None,
                                  "control": True, "expected": control["expected"]}
            payloads[judge].append(payload)
        rng.shuffle(payloads[judge])
        directory = ROOT / "judging/packets" / f"judge_{judge}"
        for start in range(0, len(payloads[judge]), 10):
            write_json(directory / f"packet_{start // 10 + 1:02d}.json", payloads[judge][start:start + 10])
    write_json(ROOT / "judging/assignment.json", assignment)
    (ROOT / "judging/instructions.md").write_text(RUBRIC)
    print({"unique_answers": 200, "second_assessments": len(double), "controls": 18,
           "ratings_per_judge": {j: len(p) for j, p in payloads.items()}}, flush=True)


def validate_rating(row, task):
    assert isinstance(row["id"], str)
    assert row["confidence"] in ["high", "medium", "low"]
    assert isinstance(row["reason"], str) and row["reason"].strip()
    if task == "continuation":
        assert row["language"] in ["bulgarian", "mixed", "other", "unclear"]
        assert all(type(row[k]) is int and row[k] in [0, 1, 2] for k in ["readability", "coherence", "relevance"])
        assert type(row["predominant_degeneration"]) is bool and type(row["uncertainty"]) is bool
    else:
        assert row["qa_grade"] in ["correct", "partial", "incorrect", "unclear"]


def resolve():
    assignment = read_json(ROOT / "judging/assignment.json")
    ratings, payloads = {}, {}
    for judge in [1, 2, 3]:
        for packet in sorted((ROOT / "judging/packets" / f"judge_{judge}").glob("packet_*.json")):
            entries = read_json(packet)
            path = ROOT / "judging/ratings" / f"judge_{judge}" / packet.name
            assessed = read_json(path)
            assert Counter(r["id"] for r in entries) == Counter(r["id"] for r in assessed), path
            for row in assessed:
                key = (row["id"], judge)
                assert key not in ratings
                validate_rating(row, assignment[row["id"]]["task"])
                ratings[key] = row
            payloads.update({p["id"]: p for p in entries})
    expected = sum(1 + bool(meta["secondary"]) for meta in assignment.values())
    assert len(ratings) == expected == 258
    controls_result, agreement, conflicts, resolved = [], [], [], []
    adjudication_path = ROOT / "judging/adjudications.json"
    adjudicated = {r["id"]: r for r in read_json(adjudication_path)} if adjudication_path.exists() else {}
    for opaque, meta in assignment.items():
        first = ratings[(opaque, meta["primary"])]
        if meta["control"]:
            controls_result.append({"judge": meta["primary"], "id": opaque,
                                    "expected": meta["expected"], "actual": acceptable(first),
                                    "passed": meta["expected"] == acceptable(first)})
            continue
        chosen = first
        if meta["secondary"]:
            second = ratings[(opaque, meta["secondary"])]
            agreement.append({"id": opaque, "task": meta["task"], "first": acceptable(first),
                              "second": acceptable(second), "exact": all(first[k] == second[k] for k in FIELDS[meta["task"]])})
            if any(first[k] != second[k] for k in FIELDS[meta["task"]]):
                conflicts.append({**payloads[opaque], "first": first, "second": second})
                if opaque in adjudicated:
                    validate_rating(adjudicated[opaque], meta["task"])
                    chosen = adjudicated[opaque]
        resolved.append({**meta, **chosen, "acceptable": acceptable(chosen)})
    write_json(ROOT / "judging/conflicts_blind.json", conflicts)
    write_json(ROOT / "judging/controls_result.json", controls_result)
    write_json(ROOT / "judging/agreement.json", agreement)
    unresolved = {r["id"] for r in conflicts} - set(adjudicated)
    if unresolved:
        raise ValueError(f"Blind adjudication required for {len(unresolved)} responses; see judging/conflicts_blind.json")
    assert set(adjudicated).issubset({r["id"] for r in conflicts})
    write_json(ROOT / "judging/resolved.json", resolved)
    return resolved, controls_result, agreement
