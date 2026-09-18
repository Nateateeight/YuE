"""Minimal tiktoken wrapper for YuE2."""
import base64
import unicodedata

class YuE2Tokenizer:
    def __init__(self, merge_file):
        import tiktoken
        ranks = {base64.b64decode(t): int(r) for t, r in
                 (line.split() for line in open(merge_file, 'rb').read().splitlines() if line)}
        specials = ["", "<|im_start|>", "<|im_end|>", "<R>", "<S>", "<X>", "<mask>", "<sep>"]
        specials += [f"<extra_{i}>" for i in range(200)]
        specials[204:206] = ["<abc>", "</abc>"]
        pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        self._enc = tiktoken.Encoding("YuE2", pat_str=pattern, mergeable_ranks=ranks,
                                     special_tokens={s: i + len(ranks) for i, s in enumerate(specials)})
    
    def encode(self, text):
        return self._enc.encode_ordinary(unicodedata.normalize("NFC", text))
    
    def decode(self, ids):
        return self._enc.decode([int(i) for i in ids if 0 <= i < self._enc.n_vocab], errors="replace")
