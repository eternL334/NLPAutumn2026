"""Cheap CPU statistics on the saved 10,000-document samples."""

import json
import os
import re
from collections import Counter
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from src.data.corpus import ARTIFACTS, SEED, write_json

GROUPS = ["raw", "hq", "top"]
WORDS = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
CYRILLIC = re.compile(r"[\u0400-\u052f\u2de0-\u2dff\ua640-\ua69f]")


def hostname(url):
    try:
        return (urlsplit(url).hostname or "unknown").lower()
    except ValueError:
        return "invalid"


def measurements(text):
    n = len(text)
    words = text.split()
    lexical = WORDS.findall(text.casefold())
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    counts = Counter(lines)
    repeated = sum((count - 1) * len(line) for line, count in counts.items())
    grams = list(zip(*(lexical[i:] for i in range(5))))
    letters = sum(char.isalpha() for char in text)
    whitespace = sum(char.isspace() for char in text)
    digits = sum(char.isdigit() for char in text)
    values = {
        "n_chars": n, "n_words": len(words), "n_lexical_words": len(lexical),
        "repeated_line_char_fraction": repeated / max(n, 1),
        "repeated_5gram_fraction": (len(grams) - len(set(grams))) / max(len(grams), 1),
        "letter_fraction": letters / max(n, 1),
        "cyrillic_letter_fraction": sum(char.isalpha() for char in CYRILLIC.findall(text)) / max(letters, 1),
        "digit_fraction": digits / max(n, 1),
        "other_symbol_fraction": (n - letters - whitespace - digits) / max(n, 1),
    }
    return values, lexical


def calculate():
    directory = ARTIFACTS / "statistics"
    directory.mkdir(exist_ok=True)
    summaries, lexical_summaries, domains = [], [], []
    for group in GROUPS:
        frame = pd.read_parquet(ARTIFACTS / "samples" / f"{group}.parquet")
        # Randomized order prevents the first source files dominating the word budget.
        frame = frame.iloc[np.random.default_rng(SEED).permutation(len(frame))]
        records, frequencies, remaining = [], Counter(), 1_000_000
        for row in frame.itertuples():
            metrics, words = measurements(row.text)
            records.append(dict(metrics, id=row.id, hostname=hostname(row.url),
                                language_score=row.language_score, quality_score=row.quality_score))
            if remaining:
                take = words[:remaining]
                frequencies.update(take)
                remaining -= len(take)
        if remaining:
            raise ValueError(f"Not enough lexical words in {group} sample")
        measured = pd.DataFrame(records)
        measured.to_parquet(directory / f"{group}_documents.parquet", index=False)
        numeric = measured.select_dtypes(include="number")
        for metric in numeric:
            values = numeric[metric].dropna()
            if len(values):
                summaries.append({"group": group, "metric": metric, "mean": float(values.mean()),
                                  "median": float(values.median()), "p90": float(values.quantile(.9)),
                                  "p99": float(values.quantile(.99))})
        by_domain = measured.groupby("hostname").agg(documents=("id", "size"), characters=("n_chars", "sum"))
        by_domain["document_share"] = by_domain.documents / len(frame)
        by_domain["character_share"] = by_domain.characters / measured.n_chars.sum()
        by_domain.sort_values("characters", ascending=False).reset_index().to_parquet(directory / f"{group}_domains.parquet", index=False)
        domains.append({"group": group, "unique_hostnames": len(by_domain),
                        "top10_document_share": float(by_domain.document_share.nlargest(10).sum()),
                        "top10_character_share": float(by_domain.character_share.nlargest(10).sum())})
        vocabulary = pd.DataFrame(frequencies.most_common(), columns=["word", "count"])
        vocabulary.to_parquet(directory / f"{group}_vocabulary.parquet", index=False)
        hapax = int((vocabulary["count"] == 1).sum())
        lexical_summaries.append({"group": group, "words": 1_000_000, "vocabulary": len(vocabulary),
                                  "hapax_types": hapax, "hapax_type_share": hapax / len(vocabulary),
                                  "hapax_token_share": hapax / 1_000_000})
        print(f"Statistics complete: {group}", flush=True)
    report = {"sample_per_group": 10_000, "seed": SEED, "metrics": summaries,
              "domains": domains, "lexical": lexical_summaries,
              "definitions": {
                  "n_words": "Whitespace-separated nonempty units (str.split); not model tokens",
                  "lexical_words": "Unicode letter words, optional internal apostrophes/hyphens, casefold",
                  "repeated_lines": "Characters in occurrences after the first of identical stripped nonempty lines / all characters",
                  "repeated_5grams": "(number of word 5-gram positions - distinct 5-grams) / positions",
                  "other_symbols": "Characters that are not letters, digits or whitespace, including normal punctuation",
                  "language_score": "Existing source classifier output; not independent language evaluation",
                  "cyrillic": "Cyrillic alphabet characters / letters; does not distinguish Bulgarian from Russian",
                  "domain": "URL hostname; no eTLD+1 aggregation"}}
    write_json(directory / "summary.json", report)
    return report


def figures():
    os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = ARTIFACTS / "figures"
    directory.mkdir(exist_ok=True)
    labels = {"raw": "FineWeb-2", "hq": "HQ-Plus", "top": "Top 1M"}
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    frames = {group: pd.read_parquet(ARTIFACTS / "statistics" / f"{group}_documents.parquet") for group in GROUPS}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for group, frame in frames.items():
        for axis, metric in zip(axes, ["n_words", "repeated_5gram_fraction"]):
            values = np.sort(frame[metric].to_numpy())
            axis.plot(values, np.arange(1, len(values) + 1) / len(values), label=labels[group])
    axes[0].set(xscale="log", xlabel="Слов в документе", ylabel="Доля документов ≤ x", title="Длины документов")
    axes[1].set(xlabel="Доля повторных 5-грамм", title="Повторяемость")
    axes[1].legend()
    fig.savefig(directory / "lengths_repetition.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for group in ["hq", "top"]:
        values = np.sort(frames[group].quality_score.to_numpy())
        axes[0].plot(values, np.arange(1, len(values) + 1) / len(values), label=labels[group])
    axes[0].set(xlabel="Опубликованный quality_score", ylabel="Доля документов ≤ x", title="Критерий отбора (не независимый эвал)")
    axes[0].legend()
    for group in GROUPS:
        vocabulary = pd.read_parquet(ARTIFACTS / "statistics" / f"{group}_vocabulary.parquet")
        axes[1].loglog(np.arange(1, len(vocabulary) + 1), vocabulary["count"], label=labels[group])
    axes[1].set(xlabel="Ранг слова", ylabel="Частота", title="Ровно 1 млн слов на корпус")
    axes[1].legend()
    fig.savefig(directory / "scores_words.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, group in zip(axes, GROUPS):
        domains = pd.read_parquet(ARTIFACTS / "statistics" / f"{group}_domains.parquet").head(10).iloc[::-1]
        axis.barh(domains.hostname, domains.character_share * 100)
        axis.set(title=labels[group], xlabel="% символов в выборке")
        axis.tick_params(axis="y", labelsize=8)
    fig.savefig(directory / "domains.png")
    plt.close(fig)

    audit_path = ARTIFACTS / "audit/summary.json"
    if audit_path.exists():
        report = json.loads(audit_path.read_text())
        intervals = pd.DataFrame(report["intervals"])
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for i, metric in enumerate(["usable", "bad_naturalness", "bad_cleanliness"]):
            rows = intervals.loc[intervals.metric == metric].set_index("group").loc[GROUPS]
            axes[i].bar([labels[group] for group in GROUPS], rows.estimate * 100,
                        yerr=np.vstack([rows.estimate - rows.low, rows.high - rows.estimate]) * 100, capsize=5)
            axes[i].set(ylabel="% документов", ylim=(0, [105, 40, 15][i]),
                        title=["Пригодность по слепому аудиту", "Серьёзные дефекты языка", "Серьёзный мусор / повреждения"][i])
        fig.savefig(directory / "audit.png")
        plt.close(fig)


if __name__ == "__main__":
    calculate()
    figures()
