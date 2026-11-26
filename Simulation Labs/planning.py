"""Grid search and continuous planning for the indoor navigation examples."""

import heapq
from collections import deque
from math import inf, log, sqrt

import numpy as np


def manhattan(a, b):
    return sum(abs(u - v) for u, v in zip(a, b))


def adjacent(p, shape):
    r, c = p
    return [
        (i, j)
        for i, j in ((r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c))
        if 0 <= i < shape[0] and 0 <= j < shape[1]
    ]


def grid_search(grid, start, goal, heuristic=manhattan):
    if grid[start] or grid[goal]:
        return [], 0
    queue = [(heuristic(start, goal), 0, start)]
    cost = {start: 0}
    parent = {start: None}
    expanded = 0
    while queue:
        _, g, u = heapq.heappop(queue)
        if cost[u] != g:
            continue
        expanded += 1
        if u == goal:
            path = []
            while u is not None:
                path.append(u)
                u = parent[u]
            return path[::-1], expanded
        for v in adjacent(u, grid.shape):
            if not grid[v] and g + 1 < cost.get(v, inf):
                cost[v] = g + 1
                parent[v] = u
                heapq.heappush(queue, (g + 1 + heuristic(v, goal), g + 1, v))
    return [], expanded


class DStarLite:
    """Reverse incremental search with lazy, versioned priority-queue entries."""

    def __init__(self, grid, start, goal):
        self.grid = np.array(grid, dtype=bool, copy=True)
        if self.grid[start] or self.grid[goal]:
            raise ValueError("Endpoints must be free")
        self.start = self.last = start
        self.goal = goal
        self.km = 0
        self.g = np.full(grid.shape, inf)
        self.rhs = self.g.copy()
        self.rhs[goal] = 0
        self.heap = []
        self.keys = {}
        self._push(goal)

    def key(self, u):
        v = min(self.g[u], self.rhs[u])
        return (v + manhattan(self.start, u) + self.km, v)

    def _push(self, u):
        k = self.key(u)
        self.keys[u] = k
        heapq.heappush(self.heap, (k, u))

    def _clean(self):
        while self.heap and self.keys.get(self.heap[0][1]) != self.heap[0][0]:
            heapq.heappop(self.heap)

    def update(self, u):
        if u != self.goal:
            self.rhs[u] = (
                inf
                if self.grid[u]
                else min(
                    (
                        1 + self.g[v]
                        for v in adjacent(u, self.grid.shape)
                        if not self.grid[v]
                    ),
                    default=inf,
                )
            )
        self.keys.pop(u, None)
        if self.g[u] != self.rhs[u]:
            self._push(u)

    def move_start(self, u):
        if self.grid[u]:
            raise ValueError("Start must be free")
        self.km += manhattan(self.last, u)
        self.last = self.start = u

    def change_cell(self, u, blocked):
        if blocked and u in (self.start, self.goal):
            raise ValueError("Cannot block an endpoint")
        self.grid[u] = blocked
        for v in [u, *adjacent(u, self.grid.shape)]:
            self.update(v)

    def repair(self):
        expanded = 0
        while True:
            self._clean()
            top = self.heap[0][0] if self.heap else (inf, inf)
            if (
                top >= self.key(self.start)
                and self.rhs[self.start] == self.g[self.start]
            ):
                break
            if not self.heap:
                raise RuntimeError("Empty queue with inconsistent start")
            old, u = heapq.heappop(self.heap)
            self.keys.pop(u, None)
            expanded += 1
            if old < self.key(u):
                self._push(u)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for v in adjacent(u, self.grid.shape):
                    self.update(v)
            else:
                self.g[u] = inf
                for v in [u, *adjacent(u, self.grid.shape)]:
                    self.update(v)
        return expanded

    def path(self):
        if not np.isfinite(self.g[self.start]):
            return []
        result = [self.start]
        while result[-1] != self.goal:
            u = result[-1]
            nxt = min(
                (v for v in adjacent(u, self.grid.shape) if not self.grid[v]),
                key=lambda v: (self.g[v], v),
            )
            if not self.g[nxt] < self.g[u]:
                raise RuntimeError("Repair the search before extracting a path")
            result.append(nxt)
        return result


def segment_free(a, b, circles):
    """Exact clearance from a line segment to each closed circular obstacle."""
    d = b - a
    length2 = d @ d
    for c, r in circles:
        t = np.clip((c - a) @ d / length2, 0, 1) if length2 else 0.0
        if np.linalg.norm(a + t * d - c) <= r:
            return False
    return True


def informed_rrt(start, goal, circles, steps=700, seed=37):
    rng = np.random.default_rng(seed)
    start = np.asarray(start, float)
    goal = np.asarray(goal, float)
    centres = [(np.asarray(c, float), r) for c, r in circles]
    if any(not segment_free(q, q, centres) for q in [start, goal]):
        raise ValueError("Endpoint in an obstacle")
    nodes = [start.copy()]
    parents = [-1]
    children = [set()]
    costs = [0.0]
    connections = set()
    distance = np.linalg.norm(goal - start)
    if not distance:
        raise ValueError("Use distinct endpoints")
    axis = (goal - start) / distance
    rotation = np.array([[axis[0], -axis[1]], [axis[1], axis[0]]])
    history = []
    best = inf
    goal_radius = 0.14
    if distance <= goal_radius and segment_free(start, goal, centres):
        connections.add(0)
        best = distance
    for _ in range(steps):
        if np.isfinite(best):
            while True:
                angle = rng.uniform(0, 2 * np.pi)
                unit = sqrt(rng.random()) * np.array([np.cos(angle), np.sin(angle)])
                radii = 0.5 * np.array([best, sqrt(max(0.0, best**2 - distance**2))])
                sample = (start + goal) / 2 + rotation @ (radii * unit)
                if np.all((sample >= 0) & (sample <= 1)):
                    break
        else:
            sample = goal if rng.random() < 0.08 else rng.uniform(0, 1, 2)
        xy = np.array(nodes)
        d = np.linalg.norm(xy - sample, axis=1)
        near = int(d.argmin())
        if d[near] > 1.0e-10:
            q = nodes[near] + (sample - nodes[near]) * min(1.0, 0.1 / d[near])
            delta = np.linalg.norm(xy - q, axis=1)
            n = len(nodes) + 1
            radius = min(0.24, 1.1 * sqrt(log(n) / n))
            neighbours = set(np.flatnonzero(delta < radius).tolist()) | {near}
            feasible = [j for j in neighbours if segment_free(nodes[j], q, centres)]
            if feasible:
                par = min(feasible, key=lambda j: costs[j] + delta[j])
                idx = len(nodes)
                nodes.append(q)
                parents.append(par)
                children.append(set())
                children[par].add(idx)
                costs.append(costs[par] + delta[par])
                for j in feasible:
                    improved = costs[idx] + delta[j]
                    if j != par and improved + 1e-12 < costs[j]:
                        change = improved - costs[j]
                        children[parents[j]].remove(j)
                        parents[j] = idx
                        children[idx].add(j)
                        queue = deque([j])
                        while queue:
                            u = queue.popleft()
                            costs[u] += change
                            queue.extend(children[u])
                if np.linalg.norm(q - goal) <= goal_radius and segment_free(
                    q, goal, centres
                ):
                    connections.add(idx)
        if connections:
            best = min(costs[j] + np.linalg.norm(nodes[j] - goal) for j in connections)
        history.append(best)
    path = []
    if connections:
        j = min(connections, key=lambda j: costs[j] + np.linalg.norm(nodes[j] - goal))
        path = [goal]
        while j != -1:
            path.append(nodes[j])
            j = parents[j]
        path.reverse()
    return np.array(path), np.asarray(history), np.asarray(nodes), parents
