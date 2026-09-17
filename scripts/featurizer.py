"""
Phase 7 - token -> 600D fixed FastText feature lookup, with caching so a
repeated token is never recomputed (guide 9.2: "FastText lookup itself does
not need GPU training. Cache token vectors so the same token is not
recomputed repeatedly.").

x_i = [ GeneralFastText(w_i) ; MedicalFastText(w_i) ]   (300D + 300D = 600D)

Production mode loads both embeddings/cc.bn.300.bin and
embeddings/medical_fasttext.bin (guide 9). A `precomputed_cache` mode is also
supported purely for lightweight local verification on machines that cannot
hold both ~5.6GB + ~2.5GB models in memory at once (see phase7_verify.py) -
on Kaggle (the guide's target platform), production mode should always be
used.
"""

import numpy as np


class FastTextFeaturizer:
    def __init__(self, general_model_path=None, medical_model_path=None, precomputed_cache=None):
        if precomputed_cache is not None:
            self.general = None
            self.medical = None
            self._cache = dict(precomputed_cache)
            self.dim = len(next(iter(self._cache.values())))
        else:
            import fasttext
            self.general = fasttext.load_model(str(general_model_path))
            self.medical = fasttext.load_model(str(medical_model_path))
            self._cache = {}
            self.dim = self.general.get_dimension() + self.medical.get_dimension()

    def vector(self, token):
        """Return the cached/looked-up 600D float32 vector for one token."""
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        if self.general is None:
            raise KeyError(
                f"Token {token!r} not in precomputed cache and no live FastText "
                "models were loaded (this featurizer was built in cache-only mode)."
            )
        g = self.general.get_word_vector(token)
        m = self.medical.get_word_vector(token)
        vec = np.concatenate([g, m]).astype(np.float32)
        self._cache[token] = vec
        return vec

    def batch_vectors(self, tokens):
        """tokens: list[str] -> np.ndarray (len(tokens), 600)."""
        return np.stack([self.vector(t) for t in tokens], axis=0)

    def cache_size(self):
        return len(self._cache)
