import gzip
import tqdm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# hyperparameters
block_size = 8  # what is the maximum context length for predictions?
# ------------


# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()[:500]

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("vocab size:", vocab_size)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Train and test splits
data = np.array(encode(text), dtype=np.int64)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_data(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # ix = np.random.randint(0, len(data) - block_size, (batch_size,))
    ix = np.array(range(len(data) - block_size))
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x, y
    return x, y


def ncd(x, x2):  # NCD with compressed lengths
    bx, bx2 = bytes(x), bytes(x2)
    x_compressed = len(gzip.compress(bx))
    x2_compressed = len(gzip.compress(bx2))
    xx2 = len(gzip.compress(bx + b" " + bx2))
    return (xx2 - min(x_compressed, x2_compressed)) / max(x_compressed, x2_compressed)


X = []
Y = []

for x, y in list(zip(*get_data("train"))):
    for token_id in range(block_size):
        context = x[:token_id + 1]
        target = y[token_id]
        # print(f"when context is {context}, target is {target}")
        X.append(context)
        Y.append(target)

print("num samples:", len(X))

train_ncd = [[ncd(X[i], X[j]) for j in range(len(X))] for i in tqdm.tqdm(range(len(X)))]


def generate(knn, context, max_new_tokens):
    print(context.shape)

    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = context[-block_size:]
        # get the predictions
        ncd_scores = np.array([ncd(idx_cond, x2) for x2 in X])
        probs = knn.predict_proba([ncd_scores])
        # sample from the distribution
        idx_next = np.random.choice(probs.shape[1], 1, p=probs[0])  # (B, 1)
        # append sampled index to the running sequence
        context = np.concatenate((context, idx_next))  # (B, T+1)
    return context


neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(train_ncd, Y)

context = np.array([0], dtype=np.int64)
print(decode(generate(neigh, context, max_new_tokens=500)))
