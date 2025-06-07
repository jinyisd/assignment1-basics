import regex as re
from collections import Counter

GPT2_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

pattern = GPT2_SPLIT_PATTERN 
vocab =  {}
merges = []
compiled_pattern = re.compile(pattern)

def get_stats(ids):
    counts = Counter()
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def train(input, vocab_size, special_tokens=None):

    merges = []

    vocab = {i : special_token for i, special_token in enumerate(special_tokens)}
    num_special_tokens = len(special_tokens)
    for i in range(256):
        vocab[num_special_tokens + i] = bytes([i])

    text = open(input, "r", encoding="utf-8").read()
    text_chunks = re.findall(compiled_pattern, text)
    ids = []
    for chunk in text_chunks:
        ids.extend(list(chunk.encode("utf-8")))

    number_merges = vocab_size - 256 - (len(special_tokens) if special_tokens else 0)

    for i in range(number_merges):
        stats = get_stats(ids)
        top_pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, top_pair, idx)
        merges.append((vocab[top_pair[0]], vocab[top_pair[1]]))
        vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
    
    return vocab, merges



