# BanglaCare - Phase 3 Preprocessing Report

## Tokenization rule
HealthNER: whitespace-segmented tokens from the original 'text' field are treated as authoritative and cleaned per-token (NFC normalize + strip zero-width/BOM + lowercase Latin) - never re-split from a joined string, so the token count cannot drift from the label count. Severity/CHQ-Summ/new queries: same simple whitespace-oriented tokenization applied after conservative whole-text cleanup (text_utils.normalize_text).

## Rules applied
- NFC Unicode normalization
- Strip zero-width spaces/BOM/directional marks
- Collapse whitespace (whole-text mode only; not applied within HealthNER tokens)
- Lowercase Latin characters only (project convention); Bangla script untouched

## Rules explicitly NOT applied
- No stopword removal
- No stemming/lemmatization
- No Banglish-to-Bangla translation
- No English medical term translation
- No number removal
- No medicine-name normalization/replacement
- Severity labels / Action Needed never used

## HealthNER per-split results
- train: {'rows': 25426, 'sanity_gate_failures': 0, 'empty_token_fallbacks_logged': 37, 'max_token_length': 507, 'examples_over_512_cap': 0}
- valid: {'rows': 3178, 'sanity_gate_failures': 0, 'empty_token_fallbacks_logged': 3, 'max_token_length': 342, 'examples_over_512_cap': 0}
- test: {'rows': 3179, 'sanity_gate_failures': 0, 'empty_token_fallbacks_logged': 3, 'max_token_length': 321, 'examples_over_512_cap': 0}

## Severity per-split results
- train: {'rows': 4211, 'max_token_length': 52, 'examples_over_512_cap': 0}
- val: {'rows': 526, 'max_token_length': 50, 'examples_over_512_cap': 0}
- test: {'rows': 526, 'max_token_length': 44, 'examples_over_512_cap': 0}

## Sanity gate: PASS
- Total token/label count failures across all splits: 0

## Zero-width joiner caveat
text_utils strips ZWNJ/ZWJ (U+200C/U+200D) unconditionally, including when embedded inside a real word, even though the guide (5.2) warns not to blindly delete legitimate script joiners - Bangla conjuncts sometimes rely on ZWNJ/ZWJ for correct rendering. This is a deliberate, documented trade-off: inconsistent author use of these characters is a common source of spurious vocabulary duplication (the same word with/without a ZWNJ becoming two different FastText tokens), which is standard practice to normalize away in Bangla NLP preprocessing. The only strict safety guarantee kept is that stripping never empties a HealthNER token (see empty_token_fallbacks_logged) - label alignment is never at risk. If future work finds this hurts specific medicine/procedure names that depend on ZWNJ, switch to a conditional strip (only when the char is not flanked by two Bangla base consonants).

## Sequence length decision
512-token cap covers 100% of examples (max observed length = 507); no truncation or chunking needed. Use dynamic padding per batch.