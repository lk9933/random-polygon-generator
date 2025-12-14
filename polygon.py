"""
Random simple polygon generation with EMST traversal and 2-opt untangling.

Algorithm:
1. Generate well-spaced points using Poisson disk sampling
2. Build Euclidean Minimum Spanning Tree via Delaunay triangulation
3. Traverse EMST with DFS to get initial vertex ordering
4. Fix self-intersections with 2-opt moves
"""

import math
import random
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from typing import Dict, List

from geometry import find_any_intersection, reverse_segment

def generate_poisson_points(n: int, min_dist: float = 2.0) -> np.ndarray:
    """
    Generates n points via Bridson's Poisson Disk Sampling. 
    Requires points be a minimum distance apart to avoid clustering.
    """
    
    cell_size = min_dist / math.sqrt(2)
    grid: Dict[tuple, int] = {}
    points = [(0.0, 0.0)]
    active = [0]
    grid[(0, 0)] = 0
    k = 30
    
    while len(points) < n and active:
        idx = random.choice(active)
        base_x, base_y = points[idx]
        found = False
        
        for _ in range(k):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(min_dist, 2 * min_dist)
            cx = base_x + math.cos(angle) * radius
            cy = base_y + math.sin(angle) * radius
            
            cell = (int(cx / cell_size), int(cy / cell_size))
            
            valid = True
            for dx in range(-2, 3):
                if not valid:
                    break
                for dy in range(-2, 3):
                    neighbor = (cell[0] + dx, cell[1] + dy)
                    if neighbor in grid:
                        px, py = points[grid[neighbor]]
                        if (cx - px) ** 2 + (cy - py) ** 2 < min_dist ** 2:
                            valid = False
                            break
            
            if valid:
                grid[cell] = len(points)
                points.append((cx, cy))
                active.append(len(points) - 1)
                found = True
                break
        
        if not found:
            active.remove(idx)
    
    return np.array(points[:n], dtype=np.float64)

def build_emst(points: np.ndarray) -> Dict[int, List[int]]:
    """Builds the Euclidean Minimum Spanning Tree of the point set."""
    n = len(points)
    
    tri = Delaunay(points)
    indptr, indices = tri.vertex_neighbor_vertices
    
    edges_i, edges_j, weights = [], [], []
    for i in range(n):
        neighbors = indices[indptr[i]:indptr[i + 1]]
        for j in neighbors:
            if j > i:
                dist = np.linalg.norm(points[i] - points[j])
                edges_i.append(i)
                edges_j.append(j)
                weights.append(dist)
    
    graph = csr_matrix((weights, (edges_i, edges_j)), shape=(n, n))
    mst = minimum_spanning_tree(graph).tocoo()
    
    adj = defaultdict(list)
    for u, v in zip(mst.row, mst.col):
        adj[u].append(v)
        adj[v].append(u)
    
    return dict(adj)

def emst_to_path(points: np.ndarray, adj: Dict[int, List[int]]) -> np.ndarray:
    """Converts EMST to vertex ordering via DFS traversal."""
    n = len(points)
    
    # Start from the leftmost point
    start = int(np.argmin(points[:, 0]))
    
    visited = [False] * n
    path_indices = []
    stack = [start]
    
    while stack:
        node = stack.pop()
        if visited[node]:
            continue
        visited[node] = True
        path_indices.append(node)
        
        for neighbor in adj.get(node, []):
            if not visited[neighbor]:
                stack.append(neighbor)
    
    return points[path_indices].copy()

def untangle_polygon(points: np.ndarray, max_iters: int = None) -> np.ndarray:
    """
    Removes self-intersections using 2-opt moves.
    2-opt move: When edges (i, i+1) and (j, j+1) cross (i < j), reverse segment [i+1, j] to uncross them.
    """
    poly = np.ascontiguousarray(points.copy())
    n = len(poly)
    
    if max_iters is None:
        max_iters = n * 100
    
    for _ in range(max_iters):
        i, j = find_any_intersection(poly)
        
        if i == -1:
            return poly
        
        reverse_segment(poly, i + 1, j)
    
    print(f"Warning: max iterations ({max_iters}) reached.")
    return poly