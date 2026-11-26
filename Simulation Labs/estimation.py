"""Planar estimation examples using known units and explicit coordinate frames."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rotation(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def icp(source, target, max_steps=60):
    moving = source.copy()
    tree = cKDTree(target)
    history = []
    R = np.eye(2)
    t = np.zeros(2)
    for _ in range(max_steps):
        _, indices = tree.query(moving)
        fixed = target[indices]
        cent = moving.mean(0)
        dest = fixed.mean(0)
        u, _, vt = np.linalg.svd((moving - cent).T @ (fixed - dest))
        d = np.ones(2)
        d[-1] = np.sign(np.linalg.det(u @ vt))
        step_R = (u * d) @ vt
        step_t = dest - cent @ step_R
        moving = moving @ step_R + step_t
        R = R @ step_R
        t = t @ step_R + step_t
        err = np.sqrt(np.mean(tree.query(moving)[0] ** 2))
        history.append(err)
        if len(history) > 1 and abs(history[-2] - err) < 1e-9:
            break
    return moving, R, t, np.array(history)


def motion(x, u, dt):
    v, w = u
    theta = x[2]
    return np.array(
        [
            x[0] + dt * v * np.cos(theta),
            x[1] + dt * v * np.sin(theta),
            wrap(theta + dt * w),
        ]
    )


def landmark_observation(x, landmark):
    d = landmark - x[:2]
    return np.array([np.linalg.norm(d), wrap(np.arctan2(d[1], d[0]) - x[2])])


def ekf_step(x, P, control, observations, landmarks, dt, control_cov, obs_cov):
    theta = x[2]
    v = control[0]
    F = np.eye(3)
    F[0, 2] = -dt * v * np.sin(theta)
    F[1, 2] = dt * v * np.cos(theta)
    G = dt * np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    x = motion(x, control, dt)
    P = F @ P @ F.T + G @ control_cov @ G.T
    for z, lm in zip(observations, landmarks):
        d = lm - x[:2]
        q = d @ d
        r = np.sqrt(q)
        if r < 1e-7:
            continue
        H = np.array([[-d[0] / r, -d[1] / r, 0], [d[1] / q, -d[0] / q, -1]])
        residual = z - landmark_observation(x, lm)
        residual[1] = wrap(residual[1])
        S = H @ P @ H.T + obs_cov
        K = np.linalg.solve(S, H @ P).T
        x = x + K @ residual
        x[2] = wrap(x[2])
        M = np.eye(3) - K @ H
        P = M @ P @ M.T + K @ obs_cov @ K.T
    return x, (P + P.T) / 2


def between(a, b):
    return np.r_[rotation(-a[2]) @ (b[:2] - a[:2]), wrap(b[2] - a[2])]


def compose(a, delta):
    return np.r_[a[:2] + rotation(a[2]) @ delta[:2], wrap(a[2] + delta[2])]


def solve_graph(initial, edges, robust=False):
    """Anchor the first pose and whiten residuals with each edge's standard deviations."""

    def residual(flat):
        states = np.vstack([initial[0], flat.reshape(-1, 3)])
        errors = []
        for i, j, measurement, std in edges:
            e = between(states[i], states[j]) - measurement
            e[2] = wrap(e[2])
            errors.extend(e / std)
        return errors

    fit = least_squares(
        residual,
        initial[1:].ravel(),
        loss="huber" if robust else "linear",
        f_scale=1.5,
        max_nfev=150,
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    result = np.vstack([initial[0], fit.x.reshape(-1, 3)])
    result[:, 2] = wrap(result[:, 2])
    return result
