import torch
import torch.nn as nn
import numpy as np

# Read and process the text file
with open('harrypotter.txt', 'r', encoding='utf-8') as file:
    data = file.read()

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# Hyperparameters
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

# Model parameters
Wxh = torch.randn(hidden_size, vocab_size, requires_grad=True) * 0.01
Whh = torch.randn(hidden_size, hidden_size, requires_grad=True) * 0.01
Why = torch.randn(vocab_size, hidden_size, requires_grad=True) * 0.01
bh = torch.zeros(hidden_size, 1, requires_grad=True)
by = torch.zeros(vocab_size, 1, requires_grad=True)

def lossFun(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = hprev
    loss = 0
    
    for t in range(len(inputs)):
        xs[t] = torch.zeros((vocab_size,1))
        xs[t][inputs[t]] = 1
        hs[t] = torch.tanh(torch.matmul(Wxh, xs[t]) + torch.matmul(Whh, hs[t-1]) + bh)
        ys[t] = torch.matmul(Why, hs[t]) + by
        ps[t] = torch.exp(ys[t]) / torch.sum(torch.exp(ys[t]))
        loss += -torch.log(ps[t][targets[t],0])
        
    dWxh, dWhh, dWhy = torch.zeros_like(Wxh), torch.zeros_like(Whh), torch.zeros_like(Why)
    dbh, dby = torch.zeros_like(bh), torch.zeros_like(by)
    dhnext = torch.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = torch.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += torch.matmul(dy, hs[t].t())
        dby += dy
        dh = torch.matmul(Why.t(), dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        dWxh += torch.matmul(dhraw, xs[t].t())
        dWhh += torch.matmul(dhraw, hs[t-1].t())
        dhnext = torch.matmul(Whh.t(), dhraw)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        torch.clamp(dparam, -5, 5, out=dparam)
        
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

n, p = 0, 0
mWxh, mWhh, mWhy = torch.zeros_like(Wxh), torch.zeros_like(Whh), torch.zeros_like(Why)
mbh, mby = torch.zeros_like(bh), torch.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size)*seq_length

while True:
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = torch.zeros((hidden_size,1))
        p = 0
    
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    
    if n % 100 == 0:
        print('iter %d, loss: %f' % (n, smooth_loss))
        
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / torch.sqrt(mem + 1e-8)
    
    p += seq_length
    n += 1

def sample(h, seed_ix, n):
    torch.no_grad()
    x = torch.zeros((vocab_size,1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = torch.tanh(torch.matmul(Wxh, x) + torch.matmul(Whh, h) + bh)
        y = torch.matmul(Why, h) + by
        p = torch.exp(y) / torch.sum(torch.exp(y))
        ix = torch.argmax(p).item()
        x = torch.zeros((vocab_size,1))
        x[ix] = 1
        ixes.append(ix)
    return ixes

h = torch.zeros((hidden_size,1))
sample_ix = sample(h, char_to_ix['H'], 200)
txt = ''.join(ix_to_char[ix] for ix in sample_ix)
print('----\n %s \n----' % (txt,))
