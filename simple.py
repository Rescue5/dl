import torch
from random import randint

def act(z):
    return torch.tanh(z)

def df(z):
    return 1 - torch.tanh(z) ** 2

def go_forward(x, w1, w2):
    z1 = torch.mv(w1[:, :3], x) + w1[:, 3]
    s1 = act(z1)

    z2 = torch.dot(w2[:2], s1) + w2[2]
    s2 = act(z2)

    return z1,z2,s1,s2  


torch.manual_seed(1)

w1 = torch.rand(8).view(2, 4)
w2 = torch.rand(3)-0.5

x_train = torch.FloatTensor([(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)])

y_train = torch.FloatTensor([-1, 1, -1, 1, -1, 1, -1, -1])

eta = 0.05
N = 1000
total = len(y_train)

for _ in range(N):
    k = randint(0, total-1)
    x = x_train[k]
    z1, z2, s1, out = go_forward(x, w1, w2)

    eps = out-y_train[k]
    delta = eps*df(z2)

    eps2 = delta * w2[:2]
    delta2 = eps2*df(z1)

    w2[:2] -= eta*delta*s1
    w2[2] -= eta*delta

    w1[0, :3] -= eta*delta2[0]*x
    w1[0, 3] -= eta*delta2[0]

    w1[1, :3] -= eta*delta2[1]*x
    w1[1, 3] -= eta*delta2[1]


for x, d in zip(x_train, y_train):
    z1, out, s1, y = go_forward(x, w1, w2)
    print(f"Выходное значение НС: {y} => {d}")

print(w1)
print(w2)