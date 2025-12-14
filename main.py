# main.py

import time
import numpy as np
import matplotlib.pyplot as plt

from polygon import generate_poisson_points, build_emst, emst_to_path, untangle_polygon
from geometry import polygon_is_simple

def plot_polygon(poly: np.ndarray, title: str, save_path: str = None):
    """Plots a simple polygon with filled interior."""
    # Ensure polygon is closed
    closed = np.vstack([poly, poly[0]])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.fill(closed[:, 0], closed[:, 1], alpha=0.35, color='steelblue',
            edgecolor='navy', linewidth=1.0)
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_simple_polygon(n: int) -> tuple:
    """Executes and times the full simple polygon generation pipeline."""
    init_time = time.perf_counter()
    
    # Generate points
    points = generate_poisson_points(n)

    # Construct EMST
    adj = build_emst(points)
    
    # Get initial ordering
    path = emst_to_path(points, adj)
    
    # Untangle crossings to simplify
    polygon = untangle_polygon(path)
    
    # Get total time
    total_time = time.perf_counter() - init_time

    return polygon, total_time

def main():
    # User input for value of n
    n = int(input("Number of vertices: ").strip())

    print(f"Generating a simple polygon with {n} vertices...")
    
    polygon, time = generate_simple_polygon(n)
    
    is_simple = polygon_is_simple(polygon)
    if not is_simple:
        raise RuntimeError(f"Unable to generate simple polygon with {n} vertices.")
    
    plot_polygon(polygon, f"Simple Polygon (n = {n}, time = {time:.3f}s)")

def generate_examples():
    examples = [10, 100, 1000, 10000]
    for n in examples:
        polygon, time = generate_simple_polygon(n)
    
        is_simple = polygon_is_simple(polygon)
        if not is_simple:
            raise RuntimeError(f"Unable to generate simple polygon with {n} vertices.")
    
        plot_polygon(polygon, f"Simple Polygon (n = {n}, time = {time:.3f}s)", f"polygon_{n}.png")

if __name__ == "__main__":
    generate_examples()
    # main()