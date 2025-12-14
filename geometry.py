# geometry.py

import numpy as np
from numba import njit
from typing import Tuple, Dict, List

@njit(cache=True)
def ccw(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    """Standard CCW orientation test."""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

@njit(cache=True)
def segments_intersect(ax: float, ay: float, bx: float, by: float,
                       cx: float, cy: float, dx: float, dy: float) -> bool:
    """Tests if segment AB intersects segment CD (excluding endpoints)."""
    d1 = ccw(ax, ay, bx, by, cx, cy)
    d2 = ccw(ax, ay, bx, by, dx, dy)
    d3 = ccw(cx, cy, dx, dy, ax, ay)
    d4 = ccw(cx, cy, dx, dy, bx, by)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)):
        if ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
    return False

@njit(cache=True)
def edges_are_adjacent(i: int, j: int, n: int) -> bool:
    """Returns True if edges i and j share a vertex."""
    i_next = (i + 1) % n
    j_next = (j + 1) % n
    return (i == j or i == j_next or i_next == j or i_next == j_next)

@njit(cache=True)
def reverse_segment(points: np.ndarray, start: int, end: int) -> None:
    """Reverses points[start:end+1] in place."""
    while start < end:
        for k in range(2):
            temp = points[start, k]
            points[start, k] = points[end, k]
            points[end, k] = temp
        start += 1
        end -= 1

@njit(cache=True)
def find_intersection_bruteforce(points: np.ndarray) -> Tuple[int, int]:
    """
    Brute force scan of all pairs of edges for intersections.
    Returns (i, j) with i < j for intersecting edges, or (-1, -1) if simple.
    """
    n = len(points)
    for i in range(n):
        i_next = (i + 1) % n
        ax, ay = points[i, 0], points[i, 1]
        bx, by = points[i_next, 0], points[i_next, 1]
        
        for j in range(i + 2, n):
            if edges_are_adjacent(i, j, n):
                continue
            
            j_next = (j + 1) % n
            cx, cy = points[j, 0], points[j, 1]
            dx, dy = points[j_next, 0], points[j_next, 1]
            
            if segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                return (i, j)
    
    return (-1, -1)

@njit(cache=True)
def polygon_is_simple(points: np.ndarray) -> bool:
    """Returns True if the polygon has no self-intersections."""
    i, j = find_intersection_bruteforce(points)
    return i == -1

class SpatialHash:
    """
    Grid-based spatial index for edge intersection queries.
    Each edge is inserted into all grid cells its bounding box overlaps.
    Queries return candidate edges sharing cells with the query edge.
    """
    
    def __init__(self, points: np.ndarray):
        self.points = points
        self.n = len(points)
        self.grid: Dict[Tuple[int, int], List[int]] = {}
        
        # Cell size = 2 x average edge length
        total_len = 0.0
        for i in range(self.n):
            j = (i + 1) % self.n
            dx = points[j, 0] - points[i, 0]
            dy = points[j, 1] - points[i, 1]
            total_len += np.sqrt(dx * dx + dy * dy)
        
        self.cell_size = max(2.0 * total_len / self.n, 1e-9)
        self.build()
    
    def to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(np.floor(x / self.cell_size)),
                int(np.floor(y / self.cell_size)))
    
    def build(self):
        """Insert all edges into the grid."""
        for i in range(self.n):
            j = (i + 1) % self.n
            x1, y1 = self.points[i]
            x2, y2 = self.points[j]
            
            c1, r1 = self.to_cell(min(x1, x2), min(y1, y2))
            c2, r2 = self.to_cell(max(x1, x2), max(y1, y2))
            
            for c in range(c1, c2 + 1):
                for r in range(r1, r2 + 1):
                    key = (c, r)
                    if key not in self.grid:
                        self.grid[key] = []
                    self.grid[key].append(i)
    
    def get_candidates(self, edge_idx: int) -> set:
        """Get edge indices that might intersect with edge_idx."""
        j = (edge_idx + 1) % self.n
        x1, y1 = self.points[edge_idx]
        x2, y2 = self.points[j]
        
        c1, r1 = self.to_cell(min(x1, x2), min(y1, y2))
        c2, r2 = self.to_cell(max(x1, x2), max(y1, y2))
        
        candidates = set()
        for c in range(c1, c2 + 1):
            for r in range(r1, r2 + 1):
                key = (c, r)
                if key in self.grid:
                    candidates.update(self.grid[key])
        
        return candidates

def find_any_intersection(points: np.ndarray) -> Tuple[int, int]:
    """
    Find any self-intersection in the polygon using spatial hashing.
    Returns (i, j) with i < j for intersecting edges, or (-1, -1) if simple.
    """
    n = len(points)
    
    hasher = SpatialHash(points)
    
    for i in range(n):
        i_next = (i + 1) % n
        ax, ay = points[i, 0], points[i, 1]
        bx, by = points[i_next, 0], points[i_next, 1]
        
        for j in hasher.get_candidates(i):
            if j <= i:
                continue
            if edges_are_adjacent(i, j, n):
                continue
            
            j_next = (j + 1) % n
            cx, cy = points[j, 0], points[j, 1]
            dx, dy = points[j_next, 0], points[j_next, 1]
            
            if segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                return (i, j)
    
    return (-1, -1)