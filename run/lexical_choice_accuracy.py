#!/usr/bin/env python3
"""
Lexical Choice Accuracy (word translation accuracy by source-word frequency).

Following ICLR 2021 "Understanding and Improving Lexical Choice in Non-Autoregressive
Translation" (Ding et al.) and RLFW-NAT ACL 2022 Table 8: for each source word
(with gold target from reference via alignment), we check whether the hypothesis
produced the correct target word at the corresponding position; report accuracy
overall and stratified by source-side word frequency (H/M/L).

Requirements:
  - Word alignment src->ref and ref->hyp (e.g. fast_align). Format: one line per
    sentence, space-separated "i-j" pairs meaning source_i -> target_j.
  - Source-side token counts to define frequency buckets (e.g. from training data
    or fairseq dict).

Usage:
  python run/lexical_choice_accuracy.py \\
    --src test.en --ref test.de --hyp test.hyp.de \\
    --align-src-ref test.en-de.align --align-ref-hyp test.de-hyp.align \\
    [--freq-file train.en.freq | --train-src train.en] \\
    [--dict-src dict.en.txt]
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path


def load_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def load_alignments(path: str) -> list[list[tuple[int, int]]]:
    """Load fast_align-style alignments. One line per sentence: 'i-j k-l ...' -> [(i,j),(k,l),...].
    i = source index, j = target index (0-indexed)."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                out.append([])
                continue
            pairs = []
            for part in line.split():
                m = re.match(r"^(\d+)-(\d+)$", part)
                if m:
                    pairs.append((int(m.group(1)), int(m.group(2))))
            out.append(pairs)
    return out


def build_src_to_ref(align_src_ref: list[tuple[int, int]]) -> dict[int, int]:
    """Map src index -> ref index. If multiple refs align to same src, keep first."""
    d = {}
    for src_i, ref_j in align_src_ref:
        if src_i not in d:
            d[src_i] = ref_j
    return d


def build_ref_to_hyp(align_ref_hyp: list[tuple[int, int]]) -> dict[int, int]:
    """Map ref index -> hyp index."""
    d = {}
    for ref_j, hyp_k in align_ref_hyp:
        if ref_j not in d:
            d[ref_j] = hyp_k
    return d


def load_freq_from_file(path: str) -> dict[str, int]:
    """File format: one line per word, 'word' or 'word\\tcount'."""
    cnt = collections.Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" in line:
                w, c = line.split("\t", 1)
                cnt[w] = int(c)
            else:
                cnt[line] += 1
    return dict(cnt)


def load_freq_from_train_src(path: str) -> dict[str, int]:
    """Count token occurrences in a tokenized corpus (one sentence per line)."""
    cnt = collections.Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            for w in line.strip().split():
                if w:
                    cnt[w] += 1
    return dict(cnt)


def load_freq_from_fairseq_dict(path: str) -> dict[str, int]:
    """Fairseq dict: 'word count' per line (count can be omitted)."""
    cnt = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().rsplit(" ", 1)
            if len(parts) == 2:
                w, c = parts[0], int(parts[1])
            else:
                w, c = parts[0], 1
            cnt[w] = c
    return cnt


def assign_buckets(
    freq: dict[str, int], frac_high: float = 1.0 / 3, frac_low: float = 1.0 / 3
) -> dict[str, str]:
    """Assign each word type to H / M / L by frequency rank. Higher count = higher rank."""
    if not freq:
        return {}
    sorted_types = sorted(freq.keys(), key=lambda w: -freq[w])
    n = len(sorted_types)
    n_high = max(1, int(n * frac_high))
    n_low = max(1, int(n * frac_low))
    bucket = {}
    for i, w in enumerate(sorted_types):
        if i < n_high:
            bucket[w] = "H"
        elif i >= n - n_low:
            bucket[w] = "L"
        else:
            bucket[w] = "M"
    return bucket


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lexical choice accuracy (word translation accuracy by src-word frequency H/M/L)."
    )
    parser.add_argument("--src", required=True, help="Source sentences (one per line, space-separated tokens)")
    parser.add_argument("--ref", required=True, help="Reference target sentences")
    parser.add_argument("--hyp", required=True, help="Model hypothesis sentences")
    parser.add_argument("--align-src-ref", required=True, help="Alignment src->ref (fast_align format)")
    parser.add_argument("--align-ref-hyp", required=True, help="Alignment ref->hyp (fast_align format)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freq-file", help="Word frequency file: one line 'word' or 'word\\tcount'")
    group.add_argument("--train-src", help="Training source corpus to count token frequencies")
    group.add_argument("--dict-src", help="Fairseq dict (word count) for source vocabulary")
    parser.add_argument("--quiet", action="store_true", help="Only print final table")
    args = parser.parse_args()

    src_lines = load_lines(args.src)
    ref_lines = load_lines(args.ref)
    hyp_lines = load_lines(args.hyp)
    align_src_ref = load_alignments(args.align_src_ref)
    align_ref_hyp = load_alignments(args.align_ref_hyp)

    n_sent = len(src_lines)
    if not (len(ref_lines) == n_sent == len(hyp_lines) == len(align_src_ref) == len(align_ref_hyp)):
        print(
            "Length mismatch: src=%d ref=%d hyp=%d align_src_ref=%d align_ref_hyp=%d"
            % (len(src_lines), len(ref_lines), len(hyp_lines), len(align_src_ref), len(align_ref_hyp)),
            file=sys.stderr,
        )
        return 1

    if args.freq_file:
        freq = load_freq_from_file(args.freq_file)
    elif args.train_src:
        freq = load_freq_from_train_src(args.train_src)
    else:
        freq = load_freq_from_fairseq_dict(args.dict_src)

    bucket = assign_buckets(freq)
    if not args.quiet:
        print("Source word types with frequency: %d; buckets H/M/L assigned." % len(bucket), file=sys.stderr)

    correct_total = 0
    total_total = 0
    correct_b = collections.defaultdict(int)
    total_b = collections.defaultdict(int)

    for idx in range(n_sent):
        src_tok = src_lines[idx].split()
        ref_tok = ref_lines[idx].split()
        hyp_tok = hyp_lines[idx].split()
        src2ref = build_src_to_ref(align_src_ref[idx])
        ref2hyp = build_ref_to_hyp(align_ref_hyp[idx])

        for src_i, ref_j in src2ref.items():
            if src_i >= len(src_tok) or ref_j >= len(ref_tok):
                continue
            hyp_k = ref2hyp.get(ref_j)
            if hyp_k is None or hyp_k >= len(hyp_tok):
                continue
            gold = ref_tok[ref_j]
            pred = hyp_tok[hyp_k]
            src_word = src_tok[src_i]
            b = bucket.get(src_word, "O")
            total_b[b] += 1
            total_total += 1
            if pred == gold:
                correct_b[b] += 1
                correct_total += 1

    # Report: overall and H / M / L (and O if present)
    def pct(c: int, t: int) -> str:
        return "%.2f" % (100.0 * c / t) if t else "N/A"

    if not args.quiet:
        print("Lexical choice accuracy (match ref word at aligned hyp position):", file=sys.stderr)
    order = ["H", "M", "L", "O"]
    for b in order:
        if total_b[b] == 0:
            continue
        acc = pct(correct_b[b], total_b[b])
        n = total_b[b]
        print("  %s: %s  (n=%d)" % (b, acc, n))
    if total_total > 0:
        print("  All: %s  (n=%d)" % (pct(correct_total, total_total), total_total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
