"""Diagnostics for the refiner (prior/refiner.py): gradient/correction sizes under the similarity loss, and a
compressed-domain MSE variant against the identity and zero baselines. Uses the first 64 pairs."""
import sys, numpy as np, torch
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prior.refiner import Refiner, prepare
z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'pairs128_0.npz'))
xs, ys = zip(*[prepare(x.astype(complex), y.astype(complex)) for x, y in zip(z['x'][:64], z['y'][:64])])
X, Y = torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
print('x abs max/rms', X.abs().max().item(), X.pow(2).mean().sqrt().item(), 'y rms', Y.pow(2).mean().sqrt().item())
torch.manual_seed(0)
m = Refiner()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
def sim(o, y): return (o*y).sum((1,2,3)) / (o.flatten(1).norm(dim=1)*y.flatten(1).norm(dim=1))
for it in range(60):
    x, y = X[it%8*8:(it%8+1)*8], Y[it%8*8:(it%8+1)*8]
    o = m(x); loss = (1-sim(o,y)).mean()
    opt.zero_grad(); loss.backward()
    g = sum(p.grad.norm()**2 for p in m.parameters()).sqrt().item()
    corr = (o-x).pow(2).mean().sqrt().item()
    opt.step()
    if it % 10 == 0: print(it, round(loss.item(),5), 'grad', g, 'corr rms', corr)
print('--- compressed-domain MSE variant')
def comp(x): return x / (1 + torch.sqrt((x ** 2).sum(1, keepdim=True)))
Xc, Yc = comp(X), comp(Y)
Xtr, Ytr, Xv, Yv = Xc[:48], Yc[:48], Xc[48:], Yc[48:]
print('identity val mse', (Xv-Yv).pow(2).mean().item(), 'zero val mse', Yv.pow(2).mean().item())
torch.manual_seed(0)
m2 = Refiner(); opt = torch.optim.Adam(m2.parameters(), lr=1e-3)
fwd = lambda x: x + m2.net(x)
for it in range(400):
    i = torch.randint(0, 48, (8,))
    loss = (fwd(Xtr[i]) - Ytr[i]).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 50 == 0:
        with torch.no_grad(): print(it, 'train', round(loss.item(),5), 'val', round((fwd(Xv)-Yv).pow(2).mean().item(),5))
