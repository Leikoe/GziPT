import gzip
import os
import time

import numpy as np
import ray
from ray.experimental import tqdm_ray
import tqdm
from sklearn.neighbors import KNeighborsClassifier
from transformers import LlamaTokenizerFast

tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")


def clearscreen(numlines=100):
    """
    Clear the console.
    numlines is an optional argument used only as a fall-back.
    """
    # Thanks to Steven D'Aprano, http://www.velocityreviews.com/forums

    if os.name == "posix":
        # Unix/Linux/MacOS/BSD/etc
        os.system('clear')
    elif os.name in ("nt", "dos", "ce"):
        # DOS/Windows
        os.system('CLS')
    else:
        # Fallback for other operating systems.
        print('\n' * numlines)


# Start Ray. This creates some processes that can do work in parallel.
ray.init(logging_level="ERROR")

# hyperparameters
n_ctx = 16  # what is the maximum context length for predictions?
n_vocab = tokenizer.vocab_size
temp = 0.800000
top_k = 40
top_p = 0.950000

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()[:5000]

# Train and test splits
data = np.array(tokenizer.encode(text), dtype=np.int64)

n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_data(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = np.array(range(0, len(data) - n_ctx))  # TODO: is step a good idea ?
    x = np.stack([data[i:i + n_ctx] for i in ix])
    y = np.stack([data[i + 1:i + n_ctx + 1] for i in ix])
    return x, y


def ncd_fast(x: bytes, x_compressed: int, x2: bytes, x2_compressed: int):  # NCD with compressed lengths
    xx2 = len(gzip.compress(b" ".join([x, x2])))
    return (xx2 - min(x_compressed, x2_compressed)) / max(x_compressed, x2_compressed)


X = []
Y = []
for x, y in list(zip(*get_data("train"))):
    for token_idx in range(n_ctx):
        context = x[:token_idx + 1]
        target = y[token_idx]
        # print(f"when context is {list(context)}, target is {target}")
        X.append([context.tobytes(), len(gzip.compress(context.tobytes()))])
        Y.append(target)


@ray.remote
def nomnom_fast(bar: tqdm_ray.tqdm, i):
    l = [ncd_fast(*X[i], *X[j]) for j in range(len(X))]
    bar.update.remote(1)
    return l


# train_ncd = [nomnom_fast(i) for i in tqdm.tqdm(range(len(X)), desc="creating model")]

remote_tqdm = ray.remote(tqdm_ray.tqdm)
bar = remote_tqdm.remote(total=len(X), desc="creating model")
train_ncd = ray.get([nomnom_fast.remote(bar, i) for i in range(len(X))])
bar.close.remote()
time.sleep(0.1)
ray.shutdown()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def inv_softmax(x, C):
    return np.log(x) + C


def generate(knn: KNeighborsClassifier, context: np.ndarray, max_new_tokens: int, streaming=False, temperature=1.0,
             top_k=None):
    if streaming:
        print(tokenizer.decode(context), end="")
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = context[-n_ctx:]
        idx_cond_compressed = len(gzip.compress(idx_cond.tobytes()))

        # get the predictions
        ncd_scores = np.array([ncd_fast(idx_cond.tobytes(), idx_cond_compressed, *x2) for x2 in X])
        probs: np.ndarray = knn.predict_proba([ncd_scores])

        # pluck the logits at the final step and scale by desired temperature
        probs = inv_softmax(probs + 0.1, 0)
        probs /= temperature
        probs = softmax(probs)
        # sample from the distribution
        if top_k is not None:
            topk_idxs = (-probs).argsort()[:top_k]
            idx_next = np.random.choice(knn.classes_[topk_idxs], 1, p=probs[0][topk_idxs])  # (B, 1)
        else:
            idx_next = np.random.choice(knn.classes_, 1, p=probs[0])  # (B, 1)
        # append sampled index to the running sequence
        context = np.concatenate((context, idx_next))  # (B, T+1)
        if streaming:
            print(tokenizer.decode(idx_next), end=" ")
    return context


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_ncd, Y)
# neigh.classes_ = np.arange(n_vocab, dtype=np.int64)

prompt = "Speak"
context = np.array(tokenizer.encode(prompt), dtype=np.int64)

print(f"n_train = {len(data)}")
print(f"n_vocab = {n_vocab}")
print(f"n_ctx   = {n_ctx}")
print()
print(f"prompt: {prompt}")
print(f"number of tokens in the prompt = {len(context)}")
for token in context:
    print(f"{token:5} -> '{tokenizer.decode(token)}'")
print()
print(f"sampling parameters: temp = {temp:6f}, top_k = {top_k}, top_p = {top_p:6f}")
print()
print()

open('output.txt', 'w').write(tokenizer.decode(generate(neigh, context, max_new_tokens=200, streaming=True)))
