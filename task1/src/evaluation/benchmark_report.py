"""Paired uncertainty estimates and reports from frozen generations and external ratings."""

import re
from collections import Counter

import numpy as np
import pandas as pd

from src.evaluation.benchmark import ROOT, read_jsonl, verify_frozen
from src.evaluation.benchmark_judging import resolve
from src.training.data import read_json, sha256, write_json

MODELS = ["monolm", "qwen"]
NAMES = {"monolm": "MonoLM-346M · 5000", "qwen": "Qwen3-0.6B-Base"}


def repeated_5grams(text):
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[i:i + 5]) for i in range(len(words) - 4)]
    return (len(grams) - len(set(grams))) / len(grams) if grams else 0.0


def uncertainty(values):
    """Rows are paired documents or QA families; columns follow MODELS."""
    rng = np.random.default_rng(42)
    samples = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return values.mean(axis=0), np.quantile(samples, [.025, .975], axis=0), np.quantile(samples[:, 0] - samples[:, 1], [.025, .975])


def agreement_summary(rows):
    if not rows:
        return {"n": 0, "agreement": None, "kappa": None}
    a = np.array([r["first"] for r in rows], dtype=float)
    b = np.array([r["second"] for r in rows], dtype=float)
    observed = float((a == b).mean())
    chance = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return {"n": len(rows), "agreement": observed,
            "kappa": (observed - chance) / (1 - chance) if chance < 1 else None,
            "exact_rubric_agreement": float(np.mean([r["exact"] for r in rows]))}


def summarize():
    from src.training.state import versions
    manifest = verify_frozen()
    resolved, controls, agreement = resolve()
    assert len(resolved) == 200 and len(agreement) == 40 and len(controls) == 18
    generated = {(model, r["id"]): r for model in MODELS
                 for r in read_jsonl(ROOT / "generations" / f"{model}.jsonl")}
    assert len(generated) == 200
    records = []
    for rating in resolved:
        output = generated[(rating["model"], rating["item_id"])]
        assert output["manifest_sha256"] == sha256(ROOT / "manifest.json")
        record = {**rating, "words": output["words"], "characters": output["characters"],
                  "empty": not output["response"].strip(), "repeat_5grams": repeated_5grams(output["response"]),
                  "seconds": output["seconds"], "response": output["response"]}
        record["strong"] = (record["task"] == "continuation" and record["acceptable"] and
                            all(record[k] == 2 for k in ["readability", "coherence", "relevance"]))
        records.append(record)
    frame = pd.DataFrame(records)
    metrics, differences = [], []

    def add(block, metric, values, answers):
        point, bounds, diff_bounds = uncertainty(values)
        for j, model in enumerate(MODELS):
            metrics.append({"block": block, "metric": metric, "model": model, "units": len(values),
                            "answers": answers, "value": float(point[j]),
                            "ci_low": float(bounds[0, j]), "ci_high": float(bounds[1, j])})
        differences.append({"block": block, "metric": metric, "comparison": "monolm_minus_qwen",
                            "value": float(point[0] - point[1]), "ci_low": float(diff_bounds[0]),
                            "ci_high": float(diff_bounds[1])})

    cont = frame[frame.task == "continuation"]
    for field, metric in [("acceptable", "minimal_acceptability"), ("strong", "all_scores_2")]:
        values = cont.pivot(index="family", columns="model", values=field)[MODELS].to_numpy(dtype=float)
        assert values.shape == (60, 2)
        add("continuation", metric, values, 60)
    qa = frame[frame.task == "qa"]
    qa_pivot = qa.pivot(index="item_id", columns="model", values="acceptable")[MODELS].sort_index()
    assert qa_pivot.shape == (40, 2)
    paired = qa_pivot.to_numpy(dtype=float).reshape(20, 2, 2)
    add("qa", "correct", paired.mean(axis=1), 40)
    add("qa", "both_paraphrases_correct", paired.min(axis=1), 40)
    automatic, defects = [], []
    for (model, task), part in frame.groupby(["model", "task"]):
        automatic.append({"model": model, "task": task, "n": len(part),
                          "mean_words": float(part.words.mean()), "mean_characters": float(part.characters.mean()),
                          "empty_rate": float(part['empty'].mean()), "repeat_5grams": float(part.repeat_5grams.mean()),
                          "mean_generation_seconds": float(part.seconds.mean())})
        if task == "continuation":
            defects.append({"model": model, "language": dict(Counter(part.language)),
                            "readability_0": int((part.readability == 0).sum()),
                            "coherence_0": int((part.coherence == 0).sum()), "relevance_0": int((part.relevance == 0).sum()),
                            "degeneration": int(part.predominant_degeneration.sum()), "uncertain": int(part.uncertainty.sum())})
        else:
            defects.append({"model": model, "qa_grades": dict(Counter(part.qa_grade))})
    controls_by_judge = [{"judge": judge, "correct": sum(r["passed"] for r in controls if r["judge"] == judge), "total": 6}
                         for judge in [1, 2, 3]]
    from src.paths import ARTIFACTS
    evidence = []
    run = ARTIFACTS / "training/runs/monolm-346m"
    for path in sorted(run.glob("metrics-*.jsonl")):
        for row in read_jsonl(path):
            if row.get("step") in [0, 5000] and "validation/loss" in row:
                evidence.append({"step": row["step"], "loss": row["validation/loss"],
                                 "tokens": row["validation/tokens"], "source": path.name})
    write_json(ROOT / "report/training_evidence.json", evidence)
    report = {"manifest_sha256": sha256(ROOT / "manifest.json"), "complete": True, "library_versions": versions(),
              "counts": {"items": 100, "generations": 200, "primary_ratings": 200, "second_ratings": 40, "control_ratings": 18},
              "metrics": metrics, "paired_differences": differences, "automatic": automatic, "defects": defects,
              "agreement": {task: agreement_summary([r for r in agreement if r["task"] == task]) for task in ["continuation", "qa"]},
              "controls": controls_by_judge, "controls_passed": all(r["passed"] for r in controls),
              "adjudications": len(read_json(ROOT / "judging/conflicts_blind.json")),
              "judge_type": "Three Codex subagents; exact backend identifier not exposed; not native-speaker ground truth",
              "gpu_budget": read_json(ROOT / "gpu_budget.json"),
              "training_evidence": evidence,
              "limitations": ["Small, selected easy QA subset; not an official Belebele score.",
                              "Validation continuations are curated for readable inputs, not a representative corpus audit.",
                              "Single sampled continuation per prompt; only sampling seed 42-derived seeds evaluated.",
                              "Models differ in size, tokenizer and pretraining; no causal attribution to our filtering.",
                              "External-model pretraining contamination is unknown; judges are model judgments.",
                              "All-zero empirical rates give degenerate [0,0] bootstrap intervals; this does not bound the population success rate at zero.",
                              "Minimum acceptability does not mean consistently coherent or factually correct writing."]}
    write_json(ROOT / "report/summary.json", report)
    frame.to_csv(ROOT / "report/ratings.csv", index=False)
    pd.DataFrame(metrics).to_csv(ROOT / "report/metrics.csv", index=False)
    pd.DataFrame(differences).to_csv(ROOT / "report/paired_differences.csv", index=False)
    items = {r["id"]: r for r in read_jsonl(ROOT / "items.jsonl")}
    examples = [{"item": items[item_id], "models": {
        model: {"response": generated[(model, item_id)]["response"],
                "rating": next(r for r in resolved if r["model"] == model and r["item_id"] == item_id)}
        for model in MODELS}} for item_id in manifest["illustrations"]]
    write_json(ROOT / "report/examples.json", examples)
    figures(report, frame)
    paths = sorted(p for p in ROOT.rglob('*') if p.is_file() and '.cache' not in p.parts and p.name != 'verification.json'
                   and p.suffix in ['.json', '.jsonl', '.md', '.csv', '.png'] and p.parent.name != 'review')
    write_json(ROOT / "verification.json", {"complete": True, "counts": report["counts"],
               "files": {str(p.relative_to(ROOT)): sha256(p) for p in paths}})
    print({"complete": True, "controls_passed": report["controls_passed"], "metrics": metrics}, flush=True)


def figures(report, frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = ROOT / "report/figures"
    directory.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(report["metrics"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, block, labels in zip(axes, ["continuation", "qa"],
                               [["Минимально приемлемо", "Все оценки 2"], ["Правильный ответ", "Обе формулировки"]]):
        keys = list(metrics[metrics.block == block].metric.unique())
        for j, model in enumerate(MODELS):
            part = metrics[(metrics.block == block) & (metrics.model == model)].set_index("metric").loc[keys]
            y = part.value.to_numpy() * 100
            error = np.vstack([y - part.ci_low.to_numpy() * 100, part.ci_high.to_numpy() * 100 - y])
            ax.bar(np.arange(2) + (j - .5) * .35, y, .35, yerr=error, capsize=3, label=NAMES[model])
        ax.set_xticks([0, 1], labels)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Доля, % · 95% bootstrap CI")
        ax.set_title("Продолжения, 60 документов" if block == "continuation" else "QA, 20 пар вопросов")
        ax.legend(fontsize=8)
    fig.supxlabel("При нулевой частоте bootstrap-интервал [0; 0] вырожден и не задаёт верхнюю границу истинной доли.", fontsize=8)
    fig.savefig(directory / "outcomes.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    cont = frame[frame.task == "continuation"]
    for ax, field, title in zip(axes, ["readability", "coherence", "relevance"], ["Понятность", "Связность", "Тема"]):
        bottom = np.zeros(2)
        for score, color in zip([0, 1, 2], ['#c85858', '#e6b85c', '#62a58b']):
            values = np.array([(cont[cont.model == model][field] == score).mean() * 100 for model in MODELS])
            ax.bar([0, 1], values, bottom=bottom, label=str(score), color=color)
            bottom += values
        ax.set_xticks([0, 1], ["MonoLM", "Qwen"])
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% продолжений")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Оценка", loc="outside lower center", ncol=3, fontsize=8)
    fig.savefig(directory / "rubric.png", dpi=150)
    plt.close(fig)
