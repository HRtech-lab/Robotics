"""Small generative policies for a two-dimensional point reaching task."""

import math
import numpy as np
import torch
from torch import nn

HORIZON = 6
ACTION_SCALE = 0.2


def expert(delta):
    """Analytic demonstrator: move 25% of the remaining displacement per tick."""
    gain = 0.25 * 0.75 ** np.arange(HORIZON)
    return delta[:, None, :] * gain[None, :, None]


class ConditionalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2 * HORIZON + 2 + 9, 96),
            nn.SiLU(),
            nn.Linear(96, 96),
            nn.SiLU(),
            nn.Linear(96, 2 * HORIZON),
        )

    def forward(self, actions, t, delta):
        frequencies = t[:, None] * actions.new_tensor([1, 2, 4, 8])[None, :] * math.pi
        time = torch.cat([t[:, None], frequencies.sin(), frequencies.cos()], dim=1)
        return self.layers(torch.cat([actions, time, delta], dim=1))


def fit_policy(kind, seed=31, steps=1250):
    """Fit either a clean-action diffusion predictor or a conditional velocity field."""
    if kind not in {"diffusion", "flow"}:
        raise ValueError("Use diffusion or flow")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-1.6, 1.6, (6000, 2)).astype(np.float32)
    actions = expert(delta) / ACTION_SCALE
    cond = torch.tensor(delta)
    data = torch.tensor(actions.reshape(len(delta), -1), dtype=torch.float32)
    model = ConditionalNet()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0018, weight_decay=1e-4)
    losses = []
    N = 32
    grid = torch.arange(N + 1, dtype=torch.float32) / N
    alpha = torch.cos((grid + 0.008) / 1.008 * math.pi / 2) ** 2
    alpha = (alpha / alpha[0]).clamp(min=0.0001, max=1.0)
    for i in range(steps):
        ids = torch.randint(0, len(data), (96,))
        clean = data[ids]
        prior = torch.randn_like(clean)
        if kind == "diffusion":
            t = torch.randint(1, N + 1, (len(ids),))
            a = alpha[t, None]
            xt = a.sqrt() * clean + (1 - a).sqrt() * prior
            output = model(xt, t.float() / N, cond[ids])
            target = clean
        else:
            t = torch.rand(len(ids))
            xt = (1 - t[:, None]) * prior + t[:, None] * clean
            output = model(xt, t, cond[ids])
            target = clean - prior
        loss = nn.functional.mse_loss(output, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if i == steps * 3 // 4:
            for group in opt.param_groups:
                group["lr"] *= 0.4
    return model.eval(), alpha, np.asarray(losses)


@torch.no_grad()
def sample(model, alpha, kind, delta, generator):
    cond = torch.as_tensor(delta, dtype=torch.float32)
    x = torch.randn((len(delta), 2 * HORIZON), generator=generator)
    if kind == "diffusion":
        steps = np.linspace(len(alpha) - 1, 0, 13, dtype=int)
        for t, prev in zip(steps, steps[1:]):
            clock = x.new_full((len(x),), float(t) / (len(alpha) - 1))
            clean = model(x, clock, cond)
            eps = (x - alpha[t].sqrt() * clean) / (1 - alpha[t]).sqrt()
            x = alpha[prev].sqrt() * clean + (1 - alpha[prev]).sqrt() * eps
    elif kind == "flow":
        n = 24
        for k in range(n):
            clock = x.new_full((len(x),), k / n)
            x = x + model(x, clock, cond) / n
    else:
        raise ValueError("Unknown policy")
    return x.reshape(len(x), HORIZON, 2).numpy() * ACTION_SCALE


def evaluate(model, alpha, kind, seed=197):
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    start = rng.uniform(-0.68, 0.68, (60, 2))
    goals = rng.uniform(-0.68, 0.68, (60, 2))
    pos = start.copy()
    paths = [pos.copy()]
    first = sample(model, alpha, kind, goals - pos, generator)
    chunk_rmse = float(np.sqrt(np.mean((first - expert(goals - pos)) ** 2)))
    for _ in range(6):
        actions = sample(model, alpha, kind, goals - pos, generator)
        for k in range(2):
            pos = pos + actions[:, k, :]
            paths.append(pos.copy())
    return np.stack(paths), goals, chunk_rmse
