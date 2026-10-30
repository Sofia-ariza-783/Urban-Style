"""Path-search algorithms on the store grid.

All algorithms share one signature,  fn(layout, start, goal, heat=None) -> SearchResult,
so the simulation can swap them freely:

    BFS               unweighted shortest path, explores in rings around the start
    A*                same optimal path, guided towards the goal by a Manhattan heuristic
    Congestion-aware  Dijkstra where busy cells cost more, so customers avoid crowds

The goal is one cell or a collection of cells (a section has several access
cells); the search stops at whichever goal is reached first, so it finds the
nearest one by walking distance. Paths include both start and the goal reached.
Unreachable goals give path=None.

Run:  python -m search.algorithms       (from task_a_pathfinding/) quick demo
"""
from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from typing import Callable, Collection

import numpy as np

import config
from store.layout import Cell, StoreLayout, load_layout


@dataclass
class SearchResult:
    path: list[Cell] | None     # start ... goal, or None if unreachable
    nodes_expanded: int         # nodes taken off the frontier and expanded
    cost: float                 # total step cost (number of steps for BFS / A*)


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _goal_set(goal: Cell | Collection[Cell]) -> frozenset[Cell]:
    """Accept a single cell or any collection of cells."""
    if isinstance(goal, tuple) and len(goal) == 2 and not isinstance(goal[0], tuple):
        return frozenset([goal])
    return frozenset(goal)


def _reconstruct(parent: dict[Cell, Cell | None], goal: Cell) -> list[Cell]:
    """Follow parent links from the goal back to the start."""
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path[::-1]


# --------------------------------------------------------------------------
# Breadth-first search
# --------------------------------------------------------------------------
def bfs(layout: StoreLayout, start: Cell, goal: Cell | Collection[Cell],
        heat: np.ndarray | None = None) -> SearchResult:
    """Breadth-first search: FIFO queue, so cells are expanded in order of
    distance from the start and the first goal reached has the fewest steps."""
    goals = _goal_set(goal)
    parent: dict[Cell, Cell | None] = {start: None}
    queue = deque([start])
    expanded = 0
    while queue:
        cell = queue.popleft()
        expanded += 1
        if cell in goals:
            path = _reconstruct(parent, cell)
            return SearchResult(path, expanded, float(len(path) - 1))
        for n in layout.neighbors(cell):
            if n not in parent:
                parent[n] = cell
                queue.append(n)
    return SearchResult(None, expanded, float("inf"))


# --------------------------------------------------------------------------
# A*
# --------------------------------------------------------------------------
def astar(layout: StoreLayout, start: Cell, goal: Cell | Collection[Cell],
          heat: np.ndarray | None = None) -> SearchResult:
    """A* with unit step costs and a Manhattan-distance heuristic.

    h(n) is the Manhattan distance from n to the nearest goal. For one goal it
    never overestimates the steps left on a 4-connected grid (admissible) and
    changes by at most 1 per step (consistent). The minimum over several goals
    keeps both properties (the minimum of consistent heuristics is consistent),
    so the path is optimal, as short as BFS's, but fewer cells are expanded.
    Frontier order: lowest f = g + h, then lowest h, then insertion order.
    """
    goals = _goal_set(goal)

    def h_cost(cell: Cell) -> int:
        return min(manhattan(cell, g) for g in goals)

    parent: dict[Cell, Cell | None] = {start: None}
    best_g = {start: 0}
    closed: set[Cell] = set()
    counter = 0
    h0 = h_cost(start)
    frontier = [(h0, h0, counter, start)]       # (f, h, insertion counter, cell)
    expanded = 0
    while frontier:
        _, _, _, cell = heapq.heappop(frontier)
        if cell in closed:                      # stale duplicate entry
            continue
        closed.add(cell)
        expanded += 1
        if cell in goals:
            return SearchResult(_reconstruct(parent, cell), expanded, float(best_g[cell]))
        g = best_g[cell] + 1
        for n in layout.neighbors(cell):
            if n not in closed and g < best_g.get(n, float("inf")):
                best_g[n] = g
                parent[n] = cell
                counter += 1
                h = h_cost(n)
                heapq.heappush(frontier, (g + h, h, counter, n))
    return SearchResult(None, expanded, float("inf"))


# --------------------------------------------------------------------------
# Congestion-aware search
# --------------------------------------------------------------------------
def dijkstra_congestion(layout: StoreLayout, start: Cell, goal: Cell | Collection[Cell],
                        heat: np.ndarray | None = None) -> SearchResult:
    """Dijkstra where entering a cell costs  1 + CONGESTION_ALPHA * h,  with
    h = that cell's heat divided by the maximum heat (so h is in [0, 1]).

    `heat` is any non-negative grid of how busy each cell is; the simulation
    passes its decaying recent-traffic grid. Crowded cells become expensive and
    the planner prefers a longer but emptier route. With heat=None (or an
    all-zero grid) every step costs 1 and this is plain Dijkstra. The search
    stops at the first goal taken off the frontier, i.e. the cheapest one.
    """
    goals = _goal_set(goal)
    if heat is not None and heat.max() > 0:
        normalised = heat / heat.max()
    else:
        normalised = np.zeros((layout.height, layout.width))

    parent: dict[Cell, Cell | None] = {start: None}
    best_cost = {start: 0.0}
    closed: set[Cell] = set()
    counter = 0
    frontier = [(0.0, counter, start)]          # (cost so far, insertion counter, cell)
    expanded = 0
    while frontier:
        cost, _, cell = heapq.heappop(frontier)
        if cell in closed:
            continue
        closed.add(cell)
        expanded += 1
        if cell in goals:
            return SearchResult(_reconstruct(parent, cell), expanded, cost)
        for n in layout.neighbors(cell):
            new_cost = cost + 1.0 + config.CONGESTION_ALPHA * normalised[n]
            if n not in closed and new_cost < best_cost.get(n, float("inf")):
                best_cost[n] = new_cost
                parent[n] = cell
                counter += 1
                heapq.heappush(frontier, (new_cost, counter, n))
    return SearchResult(None, expanded, float("inf"))


SearchFn = Callable[..., SearchResult]
ALGORITHMS: dict[str, SearchFn] = {
    "BFS": bfs,
    "A*": astar,
    "Congestion-aware": dijkstra_congestion,
}


if __name__ == "__main__":
    lay = load_layout(config.DEFAULT_LAYOUT)
    for algo_name, algo in ALGORITHMS.items():
        res = algo(lay, lay.entrance, lay.exit)
        print(f"{algo_name:<17} steps={len(res.path) - 1:>2}  nodes expanded={res.nodes_expanded}")
