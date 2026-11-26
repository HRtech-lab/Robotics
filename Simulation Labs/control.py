"""Planar tracking, bounded optimal control and arm kinematics."""

import numpy as np
from scipy.optimize import lsq_linear


def mpc_inputs(x, target, dt=0.15, horizon=15, limit=1.2):
    """Condense double-integrator dynamics into a bounded least-squares problem."""
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([0.5 * dt * dt, dt])
    transition = np.zeros((2, horizon))
    state = np.asarray(x, float)
    matrices = []
    values = []
    for k in range(horizon):
        state = A @ state
        transition = A @ transition
        transition[:, k] += B
        weights = np.sqrt([28.0, 4.0] if k == horizon - 1 else [8.0, 0.65])
        matrices.append(weights[:, None] * transition)
        values.append(weights * (target - state))
    matrices.append(np.eye(horizon) * np.sqrt(0.12))
    values.append(np.zeros(horizon))
    fit = lsq_linear(
        np.vstack(matrices), np.concatenate(values), bounds=(-limit, limit), tol=1e-8
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x


def forward_arm(q, lengths):
    angles = np.cumsum(q)
    links = lengths[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
    return np.vstack([np.zeros(2), np.cumsum(links, axis=0)])


def jacobian_arm(q, lengths):
    poses = forward_arm(q, lengths)
    vectors = poses[-1] - poses[:-1]
    return np.vstack([-vectors[:, 1], vectors[:, 0]])


def clearance(pos, obstacles, robot_radius):
    return min(
        (np.linalg.norm(pos - np.asarray(c)) - r - robot_radius for c, r in obstacles),
        default=np.inf,
    )


def unicycle_arc(pose, v, w, time):
    x, y, theta = pose
    if abs(w) < 1.0e-9:
        return np.array(
            [x + time * v * np.cos(theta), y + time * v * np.sin(theta), theta]
        )
    end = theta + w * time
    return np.array(
        [
            x + v / w * (np.sin(end) - np.sin(theta)),
            y - v / w * (np.cos(end) - np.cos(theta)),
            end,
        ]
    )


def regulated_track(path, obstacles, dt=0.06, max_steps=850):
    """Forward-only, velocity-scaled lookahead and geometric clearance regulation."""
    arcs = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    pose = np.r_[path[0] + [0, 0.04], np.arctan2(*(path[1] - path[0])[::-1])]
    track = [pose.copy()]
    speeds = []
    distances = []
    cross_track = []
    index = 0
    last = 0.35
    for _ in range(max_steps):
        dist = np.linalg.norm(path[index:] - pose[:2], axis=1)
        index += int(dist.argmin())
        cross_track.append(dist.min())
        remaining = np.linalg.norm(path[-1] - pose[:2])
        if remaining < 0.07:
            break
        look = np.clip(last * 1.25, 0.25, 0.72)
        s = min(arcs[-1], arcs[index] + look)
        carrot = np.array([np.interp(s, arcs, path[:, j]) for j in range(2)])
        delta = carrot - pose[:2]
        length = np.linalg.norm(delta)
        lateral = -np.sin(pose[2]) * delta[0] + np.cos(pose[2]) * delta[1]
        kappa = 2 * lateral / max(length**2, 1e-10)
        clear = clearance(pose[:2], obstacles, 0.1)
        curve_factor = min(1.0, 1.0 / max(abs(kappa) * 1.15, 1.0))
        proximity = np.clip(clear / 0.5, 0.2, 1.0)
        v = min(0.8 * curve_factor, 0.8 * proximity, max(0.045, remaining * 0.9))
        w = v * kappa
        projection = min(0.8, length / max(v, 1.0e-9))
        if any(
            clearance(unicycle_arc(pose, v, w, t)[:2], obstacles, 0.1) <= 0
            for t in np.linspace(0, projection, 15)
        ):
            v = w = 0.0
        speeds.append(v)
        distances.append(clear)
        pose = unicycle_arc(pose, v, w, dt)
        track.append(pose.copy())
        last = v
    return (
        np.array(track),
        np.asarray(speeds),
        np.asarray(distances),
        np.asarray(cross_track),
    )
