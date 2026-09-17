"""
Shared text-normalization and duplicate-detection utilities, used consistently
across Phase 2 (splitting), Phase 3 (preprocessing) and Phase 4 (corpus
construction) so that "the same light text normalization" in the guide means
one literal implementation, not three slightly different ones.
"""

import re
import unicodedata

BANGLA_RE = re.compile(r"[ঀ-৿]")
LATIN_RE = re.compile(r"[A-Za-z]")
ZERO_WIDTH_RE = re.compile(r"[​‌‍﻿‎‏]")


def normalize_text(text, lowercase_latin=True):
    """Canonical light normalization: NFC + strip zero-width/BOM + collapse
    whitespace + (project convention) lowercase Latin characters only.

    This single function is used for:
      - Phase 2 duplicate/near-duplicate detection (spec: "Unicode
        normalization + whitespace cleanup + lowercase Latin").
      - Phase 3/4 "light text normalization" applied to corpus lines.
    Bangla script is never altered beyond NFC composition.
    """
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = ZERO_WIDTH_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    if lowercase_latin:
        t = "".join(c.lower() if c.isascii() else c for c in t)
    return t


def clean_token(token, lowercase_latin=True):
    """Per-token cleaning for label-aligned data (HealthNER): NFC + strip
    zero-width/BOM + (project convention) lowercase Latin. Never collapses
    whitespace (a token has none) and never returns an empty string for a
    non-empty input, so token count - and therefore label alignment - is
    always preserved. If cleaning would empty the token, the original token
    is kept unchanged and the caller should log this case.
    """
    if token is None or token == "":
        return token, False
    t = unicodedata.normalize("NFC", str(token))
    t = ZERO_WIDTH_RE.sub("", t)
    if lowercase_latin:
        t = "".join(c.lower() if c.isascii() else c for c in t)
    if t == "":
        return token, True  # fallback: preserve original, flag as logged case
    return t, False


def char_shingles(text, n=5):
    """Character n-gram shingle set used for near-duplicate Jaccard similarity."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self):
        out = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            out.setdefault(r, []).append(i)
        return list(out.values())


def _bucket_by_shingle_size(shingle_sets, length_bucket):
    buckets = {}
    for idx, s in enumerate(shingle_sets):
        key = len(s) // length_bucket
        buckets.setdefault(key, []).append(idx)
    return buckets


def near_duplicate_matches_against_reference(candidate_texts, reference_texts, n=5,
                                               threshold=0.90, length_bucket=8):
    """For each candidate text, check whether it is a near-duplicate (char
    n-gram Jaccard >= threshold) of ANY reference text. Reference is indexed
    once by shingle-set-size bucket; each candidate only compares against
    reference items in its own and the adjacent size bucket, keeping cost
    roughly linear in len(candidates) * (avg reference bucket size), instead
    of the full len(candidates) * len(reference).

    Returns a set of candidate indices that matched.
    """
    ref_shingles = [char_shingles(t, n) for t in reference_texts]
    ref_buckets = _bucket_by_shingle_size(ref_shingles, length_bucket)

    matched = set()
    for i, cand in enumerate(candidate_texts):
        c_sh = char_shingles(cand, n)
        c_size = len(c_sh)
        if c_size == 0:
            continue
        key = c_size // length_bucket
        candidate_ref_idxs = ref_buckets.get(key, []) + ref_buckets.get(key - 1, []) + ref_buckets.get(key + 1, [])
        for ridx in candidate_ref_idxs:
            r_sh = ref_shingles[ridx]
            r_size = len(r_sh)
            if r_size == 0:
                continue
            if min(c_size, r_size) / max(c_size, r_size) < threshold:
                continue
            if jaccard(c_sh, r_sh) >= threshold:
                matched.add(i)
                break
    return matched


def find_near_duplicate_pairs(texts, n=5, threshold=0.90, length_bucket=8):
    """Bounded-cost near-duplicate search over a small-to-medium text list.

    Rather than comparing all O(n^2) pairs, texts are bucketed by shingle-set
    size (a Jaccard >= threshold requires very close set sizes), so only texts
    of similar length are ever compared. Pure Python/stdlib, no extra
    dependencies - appropriate for list sizes up to a few tens of thousands.

    Returns a list of (i, j, jaccard_similarity) with i < j.
    """
    shingles = [char_shingles(t, n) for t in texts]
    sizes = [len(s) for s in shingles]

    buckets = {}
    for idx, size in enumerate(sizes):
        key = size // length_bucket
        buckets.setdefault(key, []).append(idx)

    pairs = []
    seen = set()
    bucket_keys = sorted(buckets.keys())
    for key in bucket_keys:
        # compare within this bucket and the adjacent bucket (size boundary effects)
        candidate_idxs = list(buckets[key]) + list(buckets.get(key + 1, []))
        candidate_idxs = sorted(set(candidate_idxs))
        for a in range(len(candidate_idxs)):
            i = candidate_idxs[a]
            for b in range(a + 1, len(candidate_idxs)):
                j = candidate_idxs[b]
                pair_key = (i, j)
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                # cheap size-ratio prune before computing exact Jaccard
                si, sj = sizes[i], sizes[j]
                if si == 0 or sj == 0:
                    continue
                if min(si, sj) / max(si, sj) < threshold:
                    continue
                sim = jaccard(shingles[i], shingles[j])
                if sim >= threshold:
                    pairs.append((i, j, sim))
    return pairs
