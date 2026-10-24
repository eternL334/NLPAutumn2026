"""Paired comparisons, predeclared selection, plots and verified HF export."""

import os
import json

import numpy as np
import pandas as pd
from tokenizers import trainers
from src.paths import PACKAGE

from src.data.corpus import ARTIFACTS, SEED, sha256_file, write_json
from src.tokenization.experiments import (ALGORITHMS, DIRECTORY, EXAMPLES, SELECTION,
                                   SIZES, SPECIALS, candidate_name, load_tokenizer, read_json, text_batches)


def compare(names):
    rows, frequencies, paired, estimates = [], [], [], []
    corpus = read_json(ARTIFACTS / "corpus/manifest.json")
    for name in names:
        directory = DIRECTORY / "candidates" / name
        training, metrics = [read_json(directory / filename) for filename in ["training.json", "metrics.json"]]
        parameters = training["actual_vocab_size"] * SELECTION["hidden_size"]
        for split, values in metrics["splits"].items():
            frequencies.append({"candidate": name, "split": split, **values["frequency"]})
            if split == "train":
                continue
            rows.append({"candidate": name, "algorithm": name.split("_")[0], "vocab_size": training["actual_vocab_size"],
                         "split": split, "tokens": values["tokens"], "tokens_per_word": values["tokens_per_word"],
                         "characters_per_token": values["characters_per_token"], "bytes_per_token": values["bytes_per_token"],
                         **{f"length_{key}": value for key, value in values["length"].items()},
                         **{f"density_{key}": value for key, value in values["density"].items()},
                         **{f"over_{key}": value for key, value in values["longer_than_context"].items()},
                         "unk_fraction": values["unk_fraction"], "roundtrip_mismatches": values["roundtrip_mismatches"],
                         "edge_cases_passed": metrics["edge_cases"]["passed"],
                         "training_seconds": training["seconds"], "training_peak_rss_mib": training["peak_rss_mib"],
                         "encoding_mb_s": metrics["benchmark"]["utf8_mb_per_second"],
                         "encoding_tokens_s": metrics["benchmark"]["tokens_per_second"],
                         "embedding_parameters": parameters, "embedding_fraction": parameters / SELECTION["model_parameters"],
                         "embedding_bf16_mib": parameters * 2 / 2**20})
    for split in ["top_dev", "validation"]:
        frames = [pd.read_parquet(DIRECTORY / "candidates" / name / f"{split}_documents.parquet") for name in names]
        reference = frames[0]
        if not all(frame[["id", "content_hash", "n_chars"]].equals(reference[["id", "content_hash", "n_chars"]]) for frame in frames):
            raise ValueError("Paired comparison requires identical documents in identical order")
        matrix = np.column_stack([frame.n_tokens.to_numpy() for frame in frames])
        n = len(reference)
        weights = np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=2000)
        totals = weights @ matrix
        baseline = names.index("bpe_16384")
        for column, name in enumerate(names):
            absolute = totals[:, column] - totals[:, baseline]
            relative = totals[:, column] / totals[:, baseline] - 1
            paired.append({"candidate": name, "split": split, "baseline": "bpe_16384", "bootstrap_repeats": 2000,
                           "token_difference": int(matrix[:, column].sum() - matrix[:, baseline].sum()),
                           "difference_low": float(np.quantile(absolute, .025)), "difference_high": float(np.quantile(absolute, .975)),
                           "relative_difference": float(matrix[:, column].sum() / matrix[:, baseline].sum() - 1),
                           "relative_low": float(np.quantile(relative, .025)), "relative_high": float(np.quantile(relative, .975))})
        if split == "top_dev":
            characters = corpus["train_totals"]["characters"]
            bootstrap_chars = weights @ reference.n_chars.to_numpy()
            for column, name in enumerate(names):
                bootstrap_estimate = totals[:, column] / bootstrap_chars * characters
                estimates.append({"candidate": name, "content_tokens_estimate": float(matrix[:, column].sum() / reference.n_chars.sum() * characters),
                                  "low": float(np.quantile(bootstrap_estimate, .025)), "high": float(np.quantile(bootstrap_estimate, .975)),
                                  "additional_eos_tokens": corpus["train_documents"], "exact": False})
    return pd.DataFrame(rows), frequencies, paired, estimates


def select(table):
    correctness = table.assign(valid=table.unk_fraction.eq(0) & table.roundtrip_mismatches.eq(0) & table.edge_cases_passed)
    valid = correctness.groupby("candidate").valid.all()
    candidates = table.loc[table.split.eq("top_dev")].copy()
    candidates["correct"] = candidates.candidate.map(valid)
    candidates["within_budget"] = candidates.embedding_fraction.le(SELECTION["embedding_budget_fraction"])
    eligible = candidates.loc[candidates.correct & candidates.within_budget]
    if eligible.empty:
        raise ValueError("No valid tokenizer within the predeclared embedding budget")
    best = int(eligible.tokens.min())
    shortlist = eligible.loc[eligible.tokens.le(best * (1 + SELECTION["token_tolerance"]))]
    chosen = shortlist.sort_values(["vocab_size", "tokens", "algorithm"]).iloc[0]
    candidates["extra_tokens_over_best_eligible"] = candidates.tokens / best - 1
    candidates["within_token_tolerance"] = candidates.tokens.le(best * (1 + SELECTION["token_tolerance"]))
    result = {"candidate": chosen.candidate, "rule": SELECTION, "best_eligible_tokens": best,
              "selected_top_dev_tokens": int(chosen.tokens),
              "relative_extra_tokens": float(chosen.tokens / best - 1),
              "eligible": eligible.candidate.tolist(), "shortlist": shortlist.candidate.tolist(),
              "reason": "Smallest vocabulary within 5% of best eligible Top-dev token count; then fewer tokens, then BPE",
              "downstream_quality_proven": False}
    return result, candidates


def export_selected(name):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    original = load_tokenizer(name)
    destination = DIRECTORY / "selected"
    wrapped = PreTrainedTokenizerFast(tokenizer_object=original, pad_token=SPECIALS[0], unk_token=SPECIALS[1],
                                     bos_token=SPECIALS[2], eos_token=SPECIALS[3], split_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
    wrapped.save_pretrained(destination)
    loaded = AutoTokenizer.from_pretrained(destination, local_files_only=True)
    if [loaded.convert_tokens_to_ids(token) for token in SPECIALS] != list(range(4)):
        raise ValueError("Special IDs changed after HF export")
    checked, tokens = 0, 0
    for split in ["top_dev", "validation", "edge_cases"]:
        batches = [EXAMPLES] if split == "edge_cases" else text_batches(split)
        for texts in batches:
            expected = [item.ids for item in original.encode_batch_fast(texts, add_special_tokens=False)]
            actual = loaded(texts, add_special_tokens=False, padding=False, truncation=False).input_ids
            if actual != expected or loaded.batch_decode(actual, skip_special_tokens=False, clean_up_tokenization_spaces=False) != texts:
                raise ValueError(f"HF save/reload mismatch: {split}")
            checked += len(texts)
            tokens += sum(map(len, actual))
    report = {"candidate": name, "documents_and_edge_cases": checked, "checked_tokens": tokens,
              "same_ids_after_reload": True, "exact_roundtrip": True, "special_ids": dict(zip(SPECIALS, range(4))),
              "literal_special_strings_are_plain_text": True, "passed": True}
    write_json(DIRECTORY / "export_verification.json", report)
    return report


def plots(table, names):
    os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = DIRECTORY / "figures"
    directory.mkdir(exist_ok=True)
    colors = {"bpe": "#226baf", "unigram": "#d97929"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for split, axis in zip(["top_dev", "validation"], axes):
        for algorithm in ALGORITHMS:
            part = table.loc[table.split.eq(split) & table.algorithm.eq(algorithm)].sort_values("vocab_size")
            axis.plot(part.vocab_size, part.tokens_per_word, "o-", label=algorithm.upper(), color=colors[algorithm])
        axis.set(xscale="log", xlabel="Размер словаря", ylabel="Токенов на слово", title=split,
                 xticks=SIZES, xticklabels=["8k", "16k", "32k", "64k"])
        axis.minorticks_off()
        axis.legend()
    fig.savefig(directory / "compression.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    top = table.loc[table.split.eq("top_dev")]
    for algorithm in ALGORITHMS:
        part = top.loc[top.algorithm.eq(algorithm)].sort_values("vocab_size")
        axes[0].plot(part.embedding_parameters / 1e6, part.tokens / 1e6, "o-", label=algorithm.upper(), color=colors[algorithm])
        axes[1].plot(part.vocab_size, part.encoding_mb_s, "o-", label=algorithm.upper(), color=colors[algorithm])
    axes[0].axvline(SELECTION["model_parameters"] * SELECTION["embedding_budget_fraction"] / 1e6, color="grey", linestyle="--", label="Бюджет 15%")
    axes[0].set(xlabel="Embedding / выход, млн параметров", ylabel="Токенов на Top-dev, млн", title="Стоимость сжатия")
    axes[1].set(xscale="log", xlabel="Размер словаря", ylabel="UTF-8 MB/s", title="Скорость кодирования",
                xticks=SIZES, xticklabels=["8k", "16k", "32k", "64k"])
    axes[1].minorticks_off()
    for axis in axes:
        axis.legend()
    fig.savefig(directory / "cost_speed.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, split in enumerate(["top_dev", "validation"]):
        for name in names:
            frame = pd.read_parquet(DIRECTORY / "candidates" / name / f"{split}_documents.parquet")
            for axis, column in zip(axes[row], ["n_tokens", "tokens_per_1000_chars"]):
                values = np.sort(frame[column])
                axis.plot(values, np.arange(1, len(values) + 1) / len(values), label=name, linewidth=1)
                axis.set(xscale="log", ylabel="Доля документов ≤ x", xlabel=column, title=split)
    axes[0, 1].legend(fontsize=8)
    fig.savefig(directory / "length_distributions.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name in names:
        vocabulary = pd.read_parquet(DIRECTORY / "candidates" / name / "vocabulary.parquet")
        for axis, split in zip(axes, ["train", "top_dev"]):
            counts = np.sort(vocabulary.loc[vocabulary.kind.eq("learned"), f"count_{split}"].to_numpy())[::-1]
            counts = counts[counts > 0]
            axis.loglog(np.arange(1, len(counts) + 1), counts / vocabulary[f"count_{split}"].sum(), label=name, linewidth=1)
            axis.set(xlabel="Ранг обученного токена", ylabel="Доля в потоке токенов", title=split)
    axes[1].legend(fontsize=8)
    fig.savefig(directory / "token_frequencies.png", dpi=140)
    plt.close(fig)


def report():
    names = [candidate_name(algorithm, size) for algorithm in ALGORITHMS for size in SIZES]
    for name in names:
        directory = DIRECTORY / "candidates" / name
        config = read_json(directory / "config.json")
        trainer_class = trainers.BpeTrainer if config["algorithm"] == "bpe" else trainers.UnigramTrainer
        state = next(iter(json.loads(trainer_class(**config["trainer"]).__getstate__()).values()))
        state.pop("words")
        state["initial_alphabet"] = sorted(state["initial_alphabet"])
        write_json(directory / "effective_trainer.json", state)
    table, frequencies, paired, estimates = compare(names)
    selection, candidates = select(table)
    table.to_parquet(DIRECTORY / "comparison.parquet", index=False)
    candidates.to_parquet(DIRECTORY / "selection_table.parquet", index=False)
    for filename, value in [("frequencies.json", frequencies), ("paired_comparisons.json", paired),
                            ("corpus_estimates.json", estimates), ("selection.json", selection)]:
        write_json(DIRECTORY / filename, value)
    export = export_selected(selection["candidate"])
    plots(table, names)
    manifest = {"seed": SEED, "candidates": names, "samples": read_json(DIRECTORY / "samples/manifest.json"),
                "selection": selection, "export_verification": export, "test_accessed": False,
                "full_corpus_tokenized": False, "language_model_trained": False, "tokenizers_trained": len(names),
                "implementation_sha256": {name: sha256_file(PACKAGE / "tokenization" / name) for name in
                                          ["experiments.py", "metrics.py", "report.py"]},
                "limitations": ["Intrinsic tokenizer measurements do not establish downstream quality",
                                "Inference is deterministic; training bitwise reproducibility is not guaranteed by the library",
                                "CPU timings depend on machine load", "300M / width 1024 / tied weights are provisional cost assumptions"],
                "files": [{"path": str(path.relative_to(DIRECTORY)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                          for path in sorted(DIRECTORY.rglob("*")) if path.is_file() and path != DIRECTORY / "manifest.json"]}
    write_json(DIRECTORY / "manifest.json", manifest)
    print(f"Selected: {selection['candidate']}; verified HF export on {export['documents_and_edge_cases']} inputs", flush=True)


if __name__ == "__main__":
    report()
