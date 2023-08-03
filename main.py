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
ray.init()

# hyperparameters
context_size = 16  # what is the maximum context length for predictions?

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()[:500]

# here are all the unique characters that occur in this text
# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# print("vocab size:", vocab_size)

# create a mapping from characters to integers
# stoi = {ch: i for i, ch in enumerate(chars)}
# itos = {i: ch for i, ch in enumerate(chars)}
# encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
# decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Train and test splits
data = np.array(tokenizer.encode(text), dtype=np.int64)
print("training data length:", len(data))
print("vocab size:", tokenizer.vocab_size)

n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_data(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = np.array(range(0, len(data) - context_size))  # TODO: is step a good idea ?
    x = np.stack([data[i:i + context_size] for i in ix])
    y = np.stack([data[i + 1:i + context_size + 1] for i in ix])
    return x, y


def ncd_fast(x: bytes, x_compressed: int, x2: bytes, x2_compressed: int):  # NCD with compressed lengths
    xx2 = len(gzip.compress(b" ".join([x, x2])))
    return (xx2 - min(x_compressed, x2_compressed)) / max(x_compressed, x2_compressed)


X = []
Y = []
for x, y in list(zip(*get_data("train"))):
    for token_idx in range(context_size):
        context = x[:token_idx + 1]
        target = y[token_idx]
        # print(f"when context is {list(context)}, target is {target}")
        X.append([context.tobytes(), len(gzip.compress(context.tobytes()))])
        Y.append(target)

print("num samples:", len(X))


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


def generate(knn: KNeighborsClassifier, context: np.ndarray, max_new_tokens: int):
    clearscreen()
    print(tokenizer.decode(context), end="")
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = context[-context_size:]
        idx_cond_compressed = len(gzip.compress(idx_cond.tobytes()))
        # get the predictions
        ncd_scores = np.array([ncd_fast(idx_cond.tobytes(), idx_cond_compressed, *x2) for x2 in X])
        # probs = knn.predict_proba([ncd_scores])
        # sample from the distribution
        # idx_next = np.random.choice(probs.shape[1], 1, p=probs[0])  # (B, 1)
        idx_next = knn.predict([ncd_scores])
        # append sampled index to the running sequence
        context = np.concatenate((context, idx_next))  # (B, T+1)
        print(tokenizer.decode(idx_next), end=" ")
    return context


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_ncd, Y)

context = np.array([tokenizer.bos_token_id], dtype=np.int64)
print(context)
open('output.txt', 'w').write(tokenizer.decode(generate(neigh, context, max_new_tokens=200)))
