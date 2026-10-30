"""Store layout: an ASCII floor plan turned into a walkable grid.

Cells are (row, col) tuples with (0, 0) at the top-left corner. Shelves,
walls and checkout counters are blocked; customers walk on floor cells and
use the floor cells next to a shelf to "visit" its section.

Run:  python -m store.layout            (from task_a_pathfinding/) prints all layouts
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

import config

Cell = tuple[int, int]

FLOOR_CHARS = {".", "E", "X"}                      # cells customers can stand on
VALID_CHARS = FLOOR_CHARS | {"#"} | set(config.SECTIONS)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]    # up, down, left, right (4-connected)


@dataclass
class StoreLayout:
    """A rectangular store plan parsed from ASCII rows (see config.LAYOUTS)."""

    name: str
    rows: list[str]
    width: int = field(init=False)
    height: int = field(init=False)
    entrance: Cell = field(init=False)
    exit: Cell = field(init=False)
    sections: dict[str, list[Cell]] = field(init=False)   # shelf code -> shelf cells

    def __post_init__(self) -> None:
        self.height = len(self.rows)
        self.width = len(self.rows[0]) if self.rows else 0
        entrances, exits = [], []
        self.sections = {}
        for r, row in enumerate(self.rows):
            if len(row) != self.width:
                raise ValueError(f"{self.name}: row {r} has length {len(row)}, expected {self.width}")
            for c, ch in enumerate(row):
                if ch not in VALID_CHARS:
                    raise ValueError(f"{self.name}: unknown symbol {ch!r} at {(r, c)}")
                if ch == "E":
                    entrances.append((r, c))
                elif ch == "X":
                    exits.append((r, c))
                elif ch in config.SECTIONS:
                    self.sections.setdefault(ch, []).append((r, c))
        if len(entrances) != 1 or len(exits) != 1:
            raise ValueError(f"{self.name}: needs exactly one entrance 'E' and one exit 'X'")
        self.entrance, self.exit = entrances[0], exits[0]
        self._validate_reachability()

    # ----------------------------------------------------------------------
    # Grid queries
    # ----------------------------------------------------------------------
    def walkable(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width and self.rows[r][c] in FLOOR_CHARS

    def neighbors(self, cell: Cell) -> list[Cell]:
        """Walkable 4-connected neighbours of a cell."""
        r, c = cell
        candidates = ((r + dr, c + dc) for dr, dc in DIRECTIONS)
        return [n for n in candidates if self.walkable(n)]

    def walkable_cells(self) -> list[Cell]:
        return [(r, c) for r in range(self.height) for c in range(self.width)
                if self.walkable((r, c))]

    def walkable_mask(self) -> np.ndarray:
        """Boolean (height, width) array, True on walkable cells."""
        mask = np.zeros((self.height, self.width), dtype=bool)
        for r, c in self.walkable_cells():
            mask[r, c] = True
        return mask

    # ----------------------------------------------------------------------
    # Sections
    # ----------------------------------------------------------------------
    def access_cells(self, code: str) -> list[Cell]:
        """Walkable cells next to the shelves of a section (where customers stand)."""
        cells = {n for shelf in self.sections[code]
                 for dr, dc in DIRECTIONS
                 if self.walkable(n := (shelf[0] + dr, shelf[1] + dc))}
        return sorted(cells)

    def goal_cells(self, code: str) -> list[Cell]:
        """Cells that count as arriving at a target: the exit itself for 'X',
        otherwise the access cells of the section."""
        return [self.exit] if code == "X" else self.access_cells(code)

    def describe(self, cell: Cell) -> str:
        """Where a cell is: 'entrance', 'exit', the name(s) of the section(s)
        whose shelves touch it, or 'corridor'."""
        if cell == self.entrance:
            return "entrance"
        if cell == self.exit:
            return "exit"
        touching = {config.SECTIONS[code]["name"]
                    for code, shelves in self.sections.items()
                    if any(abs(cell[0] - r) + abs(cell[1] - c) == 1 for r, c in shelves)}
        return " / ".join(sorted(touching)) or "corridor"

    # ----------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------
    def reachable_from_entrance(self) -> set[Cell]:
        """Flood fill from the entrance over walkable cells."""
        seen = {self.entrance}
        queue = deque([self.entrance])
        while queue:
            for n in self.neighbors(queue.popleft()):
                if n not in seen:
                    seen.add(n)
                    queue.append(n)
        return seen

    def _validate_reachability(self) -> None:
        reachable = self.reachable_from_entrance()
        if self.exit not in reachable:
            raise ValueError(f"{self.name}: exit is not reachable from the entrance")
        for code in self.sections:
            if not reachable.intersection(self.access_cells(code)):
                raise ValueError(f"{self.name}: section '{code}' is not reachable from the entrance")


def load_layout(name: str) -> StoreLayout:
    """Build one of the layouts defined in config.LAYOUTS, checking that it
    contains every section the shopping lists can refer to."""
    if name not in config.LAYOUTS:
        raise ValueError(f"Unknown layout '{name}'. Available: {list(config.LAYOUTS)}")
    layout = StoreLayout(name, config.LAYOUTS[name])
    missing = set(config.SECTIONS) - set(layout.sections)
    if missing:
        raise ValueError(f"{name}: missing sections {sorted(missing)}")
    return layout


if __name__ == "__main__":
    for layout_name in config.LAYOUTS:
        lay = load_layout(layout_name)
        print(f"{lay.name} ({lay.width}x{lay.height}), entrance={lay.entrance}, exit={lay.exit}")
        print("\n".join(lay.rows), end="\n\n")
