import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from ortools.graph.python import min_cost_flow
import os
from typing import Tuple
import time
from scipy.spatial import cKDTree
import psutil
import resource
import multiprocessing

from utils import display_resource_monitor, extract_progress_to_process


# Rich imports
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box
console = Console()

class MemoryTracker:
    """Context manager to track memory usage during a code block."""
    
    def __init__(self, name: str = "Operation", enabled: bool = True):
        self.name = name
        self.enabled = enabled
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            # Get peak RSS from OS
            end_rusage = resource.getrusage(resource.RUSAGE_SELF)
            # maxrss is in bytes on macOS, kilobytes on Linux
            import platform
            if platform.system() == 'Darwin':  # macOS
                peak_memory = end_rusage.ru_maxrss  # already in bytes on macOS
            else:  # Linux
                peak_memory = end_rusage.ru_maxrss * 1024  # convert KB to bytes
            
            # Create Rich table for memory stats
            peak_mb = peak_memory / 1024 / 1024 
            peak_gb = peak_mb / 1024
            total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
            percentage_used = (peak_mb / total_memory_mb) * 100
            
            # Color code based on memory usage
            if percentage_used < 20.0:
                color = "green"
            elif percentage_used < 50.0:
                color = "yellow"
            else:
                color = "red"
            
            table = Table(title=f"Memory Usage: {self.name}", box=box.ROUNDED, show_header=True, header_style="bold #FF8C00")
            table.add_column("Metric", style="#FF8C00", no_wrap=True)
            table.add_column("Value", justify="right", style=color)
            
            table.add_row("Peak RAM usage", f"{peak_gb:.2f} GB")
            
            console.print()
            console.print(table)
            console.print()

class ImageTransformer:
    """
    Transform any image into a target image using optimization algorithms.
    """
    
    def __init__(self, target_path: str, weights_path: str = None, size: int = 128, track_memory: bool = False):
        """
        Initialize the transformer.
        
        Args:
            target_path: Path to the target image (e.g., Obama photo)
            weights_path: Path to weights image (grayscale, indicates pixel importance)
            size: Size to resize images to (will be square)
            track_memory: Whether to track and print memory usage
        """
        self.size = size
        self.track_memory = track_memory
        
        # Load and process target image
        target_img = Image.open(target_path).convert('RGB')
        target_img = target_img.resize((size, size), Image.Resampling.LANCZOS)
        self.target = np.array(target_img, dtype=np.float32) / 255.0
        
        # Load or create weights
        if weights_path and os.path.exists(weights_path):
            weights_img = Image.open(weights_path).convert('L')
            weights_img = weights_img.resize((size, size), Image.Resampling.LANCZOS)
            self.weights = np.array(weights_img, dtype=np.float32) / 255.0
        else:
            # Default: uniform weights
            self.weights = np.ones((size, size), dtype=np.float32)
        
        self.weights = np.expand_dims(self.weights, axis=2)
    
    def load_source_image(self, source_path: str) -> np.ndarray:
        """Load and process source image."""
        source_img = Image.open(source_path).convert('RGB')
        source_img = source_img.resize((self.size, self.size), Image.Resampling.LANCZOS)
        return np.array(source_img, dtype=np.float32) / 255.0
    
    def apply_transformation(self, source: np.ndarray, assignment: np.ndarray) -> np.ndarray:
        """
        Apply a transformation mapping to rearrange pixels.
        
        Args:
            source: Source image
            assignment: Array where assignment[i] = j means target pixel i gets source pixel j
        """
        h, w, c = source.shape
        source_flat = source.reshape(-1, c)
        result_flat = source_flat[assignment]
        return result_flat.reshape(h, w, c)
    

    def transform_mincostflow(self, source_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform using Minimum Cost Flow algorithm.
        
        The problem is modeled as:
        - Source node connects to all source pixels (supply = n)
        - All source pixels connect to all target pixels (with costs)
        - All target pixels connect to sink node (demand = n)
        """
        with MemoryTracker("Min Cost Flow", enabled=self.track_memory):
            with console.status("[#FF8C00]Loading source image...[/#FF8C00]", spinner="dots"):
                source = self.load_source_image(source_path)
            
                h, w, c = source.shape
                n_pixels = h * w
                
                # Flatten images
                source_flat = source.reshape(n_pixels, c)
                target_flat = self.target.reshape(n_pixels, c)
                weights_flat = self.weights.reshape(n_pixels, -1).mean(axis=1) * 255.0
            
            # Create the min cost flow solver
            smcf = min_cost_flow.SimpleMinCostFlow()
            
            # Node indices:
            # 0: source node
            # 1 to n_pixels: source pixels
            # n_pixels+1 to 2*n_pixels: target pixels
            # 2*n_pixels+1: sink node
            
            source_node = 0
            source_pixels_start = 1
            target_pixels_start = n_pixels + 1
            sink_node = 2 * n_pixels + 1
            
            SCALE_FACTOR = 1
            
            start_time = time.time()
            with console.status("[#FF8C00]Computing costs...[/#FF8C00]", spinner="dots"):

                # Color distance
                color_diff = target_flat[:, np.newaxis, :] - source_flat[np.newaxis, :, :]
                color_dist = np.sum(color_diff ** 2, axis=2)
                # Convert from [0,1] scale to [0,255] scale
                color_dist = color_dist * (255 ** 2)
                proximity_importance = 10.0  # Tunable parameter
                
                # NEW: Spatial distance
                target_x = np.arange(n_pixels) % w
                target_y = np.arange(n_pixels) // w
                source_x = np.arange(n_pixels) % w
                source_y = np.arange(n_pixels) // w
                
                dx = target_x[:, np.newaxis] - source_x[np.newaxis, :]
                dy = target_y[:, np.newaxis] - source_y[np.newaxis, :]
                spatial_dist = dx**2 + dy**2
                
                # Combined cost
                weighted_costs = (color_dist * weights_flat[:, np.newaxis] + 
                                (spatial_dist * proximity_importance)**2)
                # Scale and convert to integers for MCF solver
                max_cost = np.max(weighted_costs)
                if max_cost > 0:
                    normalized_costs = weighted_costs / max_cost
                    TARGET_MAX_COST = 100000  
                    int_costs = (normalized_costs * TARGET_MAX_COST).astype(np.int64)
                    int_costs = np.maximum(int_costs, 1)  # Minimum cost of 1
                else:
                    int_costs = np.ones_like(weighted_costs, dtype=np.int64)

            
            compute_time = time.time() - start_time
            console.print(f"[green]\u2713[/green] Costs computed in [bold]{compute_time:.2f}s[/bold]", highlight=False)
            
            build_start = time.time()

            parent_pid = os.getpid()
            stop_spinner_event = multiprocessing.Event()
            
            spinner_process = multiprocessing.Process(
                target=extract_progress_to_process,
                kwargs={
                    'parent_pid': parent_pid,
                    'message': "Building flow network...",
                    'stop_event': stop_spinner_event
                }
            )

            spinner_process.start()

            
            try:
                # This is the blocking call. The monitor process runs freely during this time.
                # Add edges from source node to all source pixels (capacity 1, cost 0)
                source_to_pixels_start = np.full(n_pixels, source_node, dtype=np.int32)
                source_to_pixels_end = np.arange(source_pixels_start, target_pixels_start, dtype=np.int32)
                source_to_pixels_capacity = np.ones(n_pixels, dtype=np.int32)
                source_to_pixels_cost = np.zeros(n_pixels, dtype=np.int64)
                
                smcf.add_arcs_with_capacity_and_unit_cost(
                    source_to_pixels_start,
                    source_to_pixels_end,
                    source_to_pixels_capacity,
                    source_to_pixels_cost
                )
                
                
                # Add edges from source pixels to target pixels (capacity 1, cost = distance)
                total_edges = n_pixels * n_pixels
                
                pixel_edges_start = np.repeat(np.arange(source_pixels_start, target_pixels_start, dtype=np.int32), n_pixels)
                pixel_edges_end = np.tile(np.arange(target_pixels_start, sink_node, dtype=np.int32), n_pixels)
                pixel_edges_capacity = np.ones(total_edges, dtype=np.int32)
                
                # int_costs has shape (n_targets, n_sources) where int_costs[i,j] = cost of assigning source j to target i
                # But we need costs in order: source_0→target_0, source_0→target_1, ..., source_1→target_0, ...
                # So we need to transpose
                pixel_edges_cost = int_costs.T.flatten()  # Transpose to get (n_sources, n_targets), then flatten
                
                smcf.add_arcs_with_capacity_and_unit_cost(
                    pixel_edges_start,
                    pixel_edges_end,
                    pixel_edges_capacity,
                    pixel_edges_cost
                )
                
                
                # Add edges from target pixels to sink node (capacity 1, cost 0)
                target_to_sink_start = np.arange(target_pixels_start, sink_node, dtype=np.int32)
                target_to_sink_end = np.full(n_pixels, sink_node, dtype=np.int32)
                target_to_sink_capacity = np.ones(n_pixels, dtype=np.int32)
                target_to_sink_cost = np.zeros(n_pixels, dtype=np.int64);
                
                smcf.add_arcs_with_capacity_and_unit_cost(
                    target_to_sink_start,
                    target_to_sink_end,
                    target_to_sink_capacity,
                    target_to_sink_cost
                )
                
                # Set supply and demand
                smcf.set_node_supply(source_node, n_pixels)
                smcf.set_node_supply(sink_node, -n_pixels)
            finally:
                # This block ensures the monitor ALWAYS stops, even if smcf.solve() errors out
                stop_spinner_event.set()
                spinner_process.join(timeout=1) # Wait up to 1s for it to finish cleanly
                if spinner_process.is_alive():
                    spinner_process.terminate() # Forcefully stop if it's stuck
            
            

                
            network_time = time.time() - build_start
            console.print(f"[green]\u2713[/green] Network built in [bold]{network_time:.2f}s[/bold] (total {total_edges:,} edges)", highlight=False)

            if self.track_memory:
                # --- START OF MONITORING LOGIC ---
                stop_monitor_event = multiprocessing.Event()
                parent_pid = os.getpid()
                
                monitor_process = multiprocessing.Process(
                    target=display_resource_monitor,
                    kwargs={
                        'parent_pid': parent_pid,
                        'stop_event': stop_monitor_event,
                        'show_spinner': True,
                        'message': "Solving MCF..."
                    }
                )
                
                monitor_process.start()
                
                status = -1 
                try:
                    # This is the blocking call. The monitor process runs freely during this time.
                    solve_start = time.time()
                    status = smcf.solve()
                    solve_time = time.time() - solve_start
                finally:
                    # This block ensures the monitor ALWAYS stops, even if smcf.solve() errors out
                    stop_monitor_event.set()
                    monitor_process.join(timeout=2) # Wait up to 2s for it to finish cleanly
                    if monitor_process.is_alive():
                        monitor_process.terminate() # Forcefully stop if it's stuck
                # --- END OF MONITORING LOGIC ---
                
                if status != smcf.OPTIMAL:
                    raise RuntimeError(f"Min Cost Flow solver failed with status: {status}")
            else:
                parent_pid = os.getpid()
                stop_spinner_event = multiprocessing.Event()
                
                spinner_process = multiprocessing.Process(
                    target=extract_progress_to_process,
                    kwargs={
                        'parent_pid': parent_pid,
                        'message': "Solving MCF...",
                        'stop_event': stop_spinner_event,
                        'show_time_elapsed': True
                    }
                )

                spinner_process.start()
                
                status = -1 
                try:
                    # This is the blocking call. The monitor process runs freely during this time.
                    solve_start = time.time()
                    status = smcf.solve()
                    solve_time = time.time() - solve_start
                finally:
                    # This block ensures the monitor ALWAYS stops, even if smcf.solve() errors out
                    stop_spinner_event.set()
                    spinner_process.join(timeout=1) # Wait up to 1s for it to finish cleanly
                    if spinner_process.is_alive():
                        spinner_process.terminate() # Forcefully stop if it's stuck
            
            console.print(f"[green]\u2713[/green] Solved in [bold]{solve_time:.2f}s[/bold]", highlight=False)
            
            # Extract assignment from the solution
            extract_start = time.time()
            
            assignment = np.zeros(n_pixels, dtype=np.int32)
            
            # The pixel-to-pixel arcs start after the first n_pixels arcs (source to source pixels)
            first_pixel_arc = n_pixels
            
            with Progress(
                SpinnerColumn("dots"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("[#FF8C00]Extracting assignment...", total=n_pixels)
                for source_pixel_idx in range(n_pixels):
                    # This source pixel's edges start at: first_pixel_arc + source_pixel_idx * n_pixels
                    arc_start = first_pixel_arc + source_pixel_idx * n_pixels
                    
                    # Check each of this source's n_pixels outgoing edges
                    for offset in range(n_pixels):
                        arc_idx = arc_start + offset
                        if smcf.flow(arc_idx) > 0:
                            target_pixel_idx = offset
                            assignment[target_pixel_idx] = source_pixel_idx
                            break  # Each source sends to exactly one target
                    
                    if source_pixel_idx % 100 == 0:
                        progress.update(task, advance=100)
            
            extract_time = time.time() - extract_start
            console.print(f"[green]\u2713[/green] Assignment extracted in [bold]{extract_time:.2f}s[/bold]", highlight=False)
            
            result = self.apply_transformation(source, assignment)
            
            return assignment, result
    
    def transform_sparse_mcf(self, source_path: str, k_neighbors: int = 100, add_slack: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform using Sparse Min Cost Flow - only connect each target pixel to its
        k nearest source pixels in combined color+spatial space. This dramatically reduces memory usage
        while staying very close to optimal.
        
        Args:
            source_path: Path to source image
            k_neighbors: Number of nearest neighbors to consider per target pixel
            add_slack: If True, add high-cost edges to ensure feasibility
        
        Memory: O(k*n) instead of O(n^2)
        """
        with MemoryTracker("Sparse Min Cost Flow", enabled=self.track_memory):
            with console.status("[#FF8C00]Loading source image...[/#FF8C00]", spinner="dots"):
                source = self.load_source_image(source_path)
            
            h, w, c = source.shape
            n_pixels = h * w
            
            # Flatten images
            source_flat = source.reshape(n_pixels, c)
            target_flat = self.target.reshape(n_pixels, c)
            weights_flat = self.weights.reshape(n_pixels, -1).mean(axis=1) * 255.0
            
            # AUTO-SWITCH: If k is too large, fall back to dense method
            density_ratio = k_neighbors / n_pixels
            if density_ratio > 0.5:  # If using >50% of edges, dense is faster
                console.print(f"[yellow]k={k_neighbors} is large relative to n={n_pixels} (ratio={density_ratio:.1%})[/yellow]", highlight=False)
                console.print(f"  [yellow]Switching to dense Min Cost Flow for better performance...[/yellow]")
                return self.transform_mincostflow(source_path)
            
            with console.status("[#FF8C00]Setting up Sparse Min Cost Flow for {n_pixels:,} pixels...[/#FF8C00]", spinner="dots"):
                
                # Build combined color+spatial feature space for k-NN search
                proximity_importance = 10.0
                
                # Create 5D feature vectors: [R, G, B, x*proximity, y*proximity]
                # Scale spatial coordinates to match color importance
                source_x = np.arange(n_pixels) % w
                source_y = np.arange(n_pixels) // w
                source_features = np.column_stack([
                    source_flat * 255.0,  # RGB in [0, 255] scale
                    (source_x * proximity_importance) ** 2,  
                    (source_y * proximity_importance) ** 2   
                ])
                
                target_x = np.arange(n_pixels) % w
                target_y = np.arange(n_pixels) // w
                target_features = np.column_stack([
                    target_flat * 255.0,  # RGB in [0, 255] scale
                    (target_x * proximity_importance) ** 2,  
                    (target_y * proximity_importance) ** 2   
                ])
                
                # Build KD-tree in combined feature transformationpace
                kdtree = cKDTree(source_features)

                
                # For each target pixel, find k nearest source pixels in combined space
                
                query_k = min(k_neighbors, n_pixels)
                distances, neighbor_indices = kdtree.query(target_features, k=query_k, workers=-1)
                
                # If k > n_pixels, we already have all pixels
                if query_k < k_neighbors:
                    console.print(f"  [dim]Note: k_neighbors={k_neighbors} > n_pixels={n_pixels}, using all pixels[/dim]", highlight=False)
                    k_neighbors = n_pixels
                
                
                # Create the min cost flow solver
                smcf = min_cost_flow.SimpleMinCostFlow()
                
                source_node = 0
                source_pixels_start = 1
                target_pixels_start = n_pixels + 1
                sink_node = 2 * n_pixels + 1
                
                SCALE_FACTOR = 1
                
            build_start = time.time()
                


            with console.status("[#FF8C00]Building sparse network...[/#FF8C00]", spinner="dots"):
                
                source_to_pixels_start = np.full(n_pixels, source_node, dtype=np.int32)
                source_to_pixels_end = np.arange(source_pixels_start, target_pixels_start, dtype=np.int32)
                source_to_pixels_capacity = np.ones(n_pixels, dtype=np.int32)
                source_to_pixels_cost = np.zeros(n_pixels, dtype=np.int64)
                
                smcf.add_arcs_with_capacity_and_unit_cost(
                    source_to_pixels_start,
                    source_to_pixels_end,
                    source_to_pixels_capacity,
                    source_to_pixels_cost
                )
               
                
                # Add sparse edges from source pixels to target pixels
                
                target_indices = np.repeat(np.arange(n_pixels), k_neighbors)
                source_indices = neighbor_indices.flatten()
                
                
                target_colors = target_flat[target_indices]
                source_colors = source_flat[source_indices]
                color_diff = target_colors - source_colors
                color_dist = np.sum(color_diff ** 2, axis=1)
                color_dist = color_dist * (255 ** 2)
                
                # Spatial distance
                target_x_coords = target_indices % w
                target_y_coords = target_indices // w
                source_x_coords = source_indices % w
                source_y_coords = source_indices // w
                dx = target_x_coords - source_x_coords
                dy = target_y_coords - source_y_coords
                spatial_dist = dx**2 + dy**2
                
                # Combined cost
                target_weights = weights_flat[target_indices]
                weighted_costs = (color_dist * target_weights + 
                                 (spatial_dist * proximity_importance)**2)
                cost_ints = (weighted_costs * SCALE_FACTOR).astype(np.int64)
                
                # Fast duplicate removal using sorting
                # Create edge identifiers as single integers for faster comparison
                edge_ids = source_indices.astype(np.int64) * n_pixels + target_indices.astype(np.int64)
                
                # Sort by edge_id to group duplicates together
                sort_indices = np.argsort(edge_ids)
                edge_ids_sorted = edge_ids[sort_indices]
                unique_mask = np.concatenate([[True], edge_ids_sorted[1:] != edge_ids_sorted[:-1]])
                unique_sorted_indices = sort_indices[unique_mask]
                
                # Keep only unique edges
                source_indices = source_indices[unique_sorted_indices]
                target_indices = target_indices[unique_sorted_indices]
                cost_ints = cost_ints[unique_sorted_indices]
                
                max_cost_seen = np.max(cost_ints) if len(cost_ints) > 0 else SCALE_FACTOR
                
                # Convert to node indices
                sparse_starts = (source_pixels_start + source_indices).astype(np.int32)
                sparse_ends = (target_pixels_start + target_indices).astype(np.int32)
                sparse_costs = cost_ints
               
                
                # Add slack edges to ensure feasibility
                if add_slack:
                    slack_cost = int(max_cost_seen * 10)
                    
                    # Instead of random slack, ensure each target has at least one edge
                    # Check which targets might be underconnected
                    target_edge_counts = np.bincount(target_indices, minlength=n_pixels)
                    underconnected = np.where(target_edge_counts < k_neighbors // 2)[0]
                    
                    if len(underconnected) > 0:
                        # For underconnected targets, add random connections
                        n_backup_per_target = max(3, k_neighbors // 10)
                        backup_targets = np.repeat(underconnected, n_backup_per_target)
                        backup_sources = np.random.randint(0, n_pixels, size=len(backup_targets), dtype=np.int32)
                    else:
                        # Add minimal random backup edges for robustness
                        np.random.seed(42)
                        n_backup = max(3, k_neighbors // 50)  
                        backup_targets = np.repeat(np.arange(n_pixels), n_backup)
                        backup_sources = np.random.randint(0, n_pixels, size=len(backup_targets), dtype=np.int32)
                    
                    # Create edge IDs for quick comparison
                    backup_edge_ids = backup_sources.astype(np.int64) * n_pixels + backup_targets.astype(np.int64)
                    existing_edge_ids = source_indices.astype(np.int64) * n_pixels + target_indices.astype(np.int64)
                    
                    # Existing edges for binary search (already sorted)
                    existing_sorted = existing_edge_ids
                    
                    # Use searchsorted to find which backup edges already exist
                    positions = np.searchsorted(existing_sorted, backup_edge_ids)
                    # Check if the backup edge actually exists at that position
                    valid_positions = positions < len(existing_sorted)
                    matches = np.zeros(len(backup_edge_ids), dtype=bool)
                    matches[valid_positions] = (existing_sorted[positions[valid_positions]] == backup_edge_ids[valid_positions])
                    
                    # Keep only backup edges that don't exist
                    new_mask = ~matches
                    backup_sources = backup_sources[new_mask]
                    backup_targets = backup_targets[new_mask]
                    
                    if len(backup_sources) > 0:
                        backup_starts = (source_pixels_start + backup_sources).astype(np.int32)
                        backup_ends = (target_pixels_start + backup_targets).astype(np.int32)
                        backup_costs = np.full(len(backup_starts), slack_cost, dtype=np.int64)
                        
                        # Append slack edges
                        sparse_starts = np.concatenate([sparse_starts, backup_starts])
                        sparse_ends = np.concatenate([sparse_ends, backup_ends])
                        sparse_costs = np.concatenate([sparse_costs, backup_costs])
                
               
                
                total_edges = len(sparse_starts)
                sparse_capacity = np.ones(total_edges, dtype=np.int32)
                
                
                smcf.add_arcs_with_capacity_and_unit_cost(
                    sparse_starts,
                    sparse_ends,
                    sparse_capacity,
                    sparse_costs
                )
               
                
                # Add edges from target pixels to sink
                
                target_to_sink_start = np.arange(target_pixels_start, sink_node, dtype=np.int32)
                target_to_sink_end = np.full(n_pixels, sink_node, dtype=np.int32)
                target_to_sink_capacity = np.ones(n_pixels, dtype=np.int32)
                target_to_sink_cost = np.zeros(n_pixels, dtype=np.int64);
                
                smcf.add_arcs_with_capacity_and_unit_cost(
                    target_to_sink_start,
                    target_to_sink_end,
                    target_to_sink_capacity,
                    target_to_sink_cost
                )
                
                # Set supply and demand
                smcf.set_node_supply(source_node, n_pixels)
                smcf.set_node_supply(sink_node, -n_pixels)
                
            network_time = time.time() - build_start
            console.print(f"[green]\u2713[/green] Sparse network built in [bold]{network_time:.2f}s[/bold] ({total_edges:,} unique edges)", highlight=False)
            
            if self.track_memory:
                # --- START OF MONITORING LOGIC ---
                stop_monitor_event = multiprocessing.Event()
                parent_pid = os.getpid()
                
                monitor_process = multiprocessing.Process(
                    target=display_resource_monitor,
                    kwargs={
                        'parent_pid': parent_pid,
                        'stop_event': stop_monitor_event,
                        'show_spinner': True,
                        'message': "Solving sparse MCF..."
                    }
                )
                
                monitor_process.start()
                
                status = -1 
                try:
                    # This is the blocking call. The monitor process runs freely during this time.
                    solve_start = time.time()
                    status = smcf.solve()
                    solve_time = time.time() - solve_start
                finally:
                    # This block ensures the monitor ALWAYS stops, even if smcf.solve() errors out
                    stop_monitor_event.set()
                    monitor_process.join(timeout=2) # Wait up to 2s for it to finish cleanly
                    if monitor_process.is_alive():
                        monitor_process.terminate() # Forcefully stop if it's stuck
                # --- END OF MONITORING LOGIC ---
            else:
                parent_pid = os.getpid()
                stop_spinner_event = multiprocessing.Event()
                
                spinner_process = multiprocessing.Process(
                    target=extract_progress_to_process,
                    kwargs={
                        'parent_pid': parent_pid,
                        'message': "Solving sparse MCF...",
                        'stop_event': stop_spinner_event,
                        'show_time_elapsed': True
                    }
                )

                spinner_process.start()
                
                status = -1 
                try:
                    # This is the blocking call. The monitor process runs freely during this time.
                    solve_start = time.time()
                    status = smcf.solve()
                    solve_time = time.time() - solve_start
                finally:
                    # This block ensures the monitor ALWAYS stops, even if smcf.solve() errors out
                    stop_spinner_event.set()
                    spinner_process.join(timeout=1) # Wait up to 1s for it to finish cleanly
                    if spinner_process.is_alive():
                        spinner_process.terminate() # Forcefully stop if it's stuck
            
            if status != smcf.OPTIMAL:
                raise RuntimeError(f"Sparse Min Cost Flow solver failed with status: {status}")
            
            console.print(f"[green]\u2713[/green] Solved in [bold]{solve_time:.2f}s[/bold]", highlight=False)
            
            # Extract assignment
            extract_start = time.time()
            assignment = np.zeros(n_pixels, dtype=np.int32)
            
            # Check only the sparse pixel-to-pixel arcs
            first_pixel_arc = n_pixels
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("[#FF8C00]Extracting assignment...", total=total_edges)
                for arc_idx in range(first_pixel_arc, first_pixel_arc + total_edges):
                    if smcf.flow(arc_idx) > 0:
                        tail = smcf.tail(arc_idx)
                        head = smcf.head(arc_idx)
                        source_pixel_idx = tail - source_pixels_start
                        target_pixel_idx = head - target_pixels_start
                        assignment[target_pixel_idx] = source_pixel_idx
                    
                    if arc_idx % 1000 == 0:
                        progress.update(task, advance=1000)
                
                # Update remaining
                progress.update(task, completed=total_edges)

            extract_time = time.time() - extract_start
            console.print(f"[green]\u2713[/green] Assignment extracted in [bold]{extract_time:.2f}s[/bold]", highlight=False)
            result = self.apply_transformation(source, assignment)
            
            return assignment, result
    
    def transform_hungarian(self, source_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform using Hungarian algorithm (optimal assignment).
        This finds the optimal one-to-one pixel mapping.

        Slow, but less memory than dense MCF.
        """
        with MemoryTracker("Hungarian Algorithm", enabled=self.track_memory):
            with console.status("[#FF8C00]Loading source image...[/#FF8C00]", spinner="dots"):
                source = self.load_source_image(source_path)
            
            h, w, c = source.shape
            n_pixels = h * w
            
            source_flat = source.reshape(n_pixels, c)
            target_flat = self.target.reshape(n_pixels, c)
            weights_flat = self.weights.reshape(n_pixels, -1).mean(axis=1) * 255.0
            
            
            
            
            start_time = time.time()

            # Compute cost matrix with spatial component
            with console.status("[#FF8C00]Computing cost matrix...[/#FF8C00]", spinner="dots"):
                cost_matrix = self.compute_cost_matrix_with_spatial(source_flat, target_flat, weights_flat, w, proximity_importance=10.0)
            
            compute_time = time.time() - start_time
            console.print(f"[green]\u2713[/green] Cost matrix computed in [bold]{compute_time:.2f}s[/bold]", highlight=False)
            
            # Run Hungarian with spinner
            
            if self.track_memory:
                # --- START OF MONITORING LOGIC ---
                stop_monitor_event = multiprocessing.Event()
                parent_pid = os.getpid()
                
                monitor_process = multiprocessing.Process(
                    target=display_resource_monitor,
                    kwargs={
                        'parent_pid': parent_pid,
                        'stop_event': stop_monitor_event,
                        'show_spinner': True,
                        'message': "Solving with Hungarian algorithm..."
                    }
                )
                
                monitor_process.start()
                
                status = -1 
                try:
                    # This is the blocking call. The monitor process runs freely during this time.
                    solve_start = time.time()
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    solve_time = time.time() - solve_start

                finally:
                    # This block ensures the monitor ALWAYS stops, even if smcf.solve() errors out
                    stop_monitor_event.set()
                    monitor_process.join(timeout=2) # Wait up to 2s for it to finish cleanly
                    if monitor_process.is_alive():
                        monitor_process.terminate() # Forcefully stop if it's stuck
                # --- END OF MONITORING LOGIC ---
            else:
                solve_start = time.time()
                spinner = Progress(
                SpinnerColumn("dots", style="#FF8C00"),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
                transient=True
            )
                spinner.add_task(description="[#FF8C00]Solving with Hungarian algorithm...[/#FF8C00]", total=None)
                with spinner:
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                solve_time = time.time() - solve_start

            console.print(f"[green]\u2713[/green] Solved in [bold]{solve_time:.2f}s[/bold]", highlight=False)

            # col_ind[i] tells us which source pixel should go to target position i
            assignment = col_ind
            
            result = self.apply_transformation(source, assignment)
            
            return assignment, result
    
    def transform_greedy_swap(self, source_path: str, max_distance: float = None, swaps_per_generation: int = None, max_generations: int = 1000, proximity_importance: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform using Greedy Swap Algorithm - iteratively swap pixels to improve assignment.
        This only considers local swaps within a shrinking neighborhood.

        Memory: O(n) - very low memory usage
        Speed: Fast
        Quality: Good local optimum, may not be globally optimal
        """
        with MemoryTracker("Greedy Swap Algorithm", enabled=self.track_memory):
            with console.status("[#FF8C00]Loading source image...[/#FF8C00]", spinner="dots"):
                source = self.load_source_image(source_path)
            
            h, w, c = source.shape
            n_pixels = h * w

            if swaps_per_generation is None:
                # Aim for ~10-20x the number of pixels for good coverage
                swaps_per_generation = max(10000, min(500000, n_pixels * 15))
        
            
            source_flat = source.reshape(n_pixels, c)
            target_flat = self.target.reshape(n_pixels, c)
            weights_flat = self.weights.reshape(n_pixels, -1).mean(axis=1) * 255.0
            
            if max_distance is None:
                max_distance = w  # Start with full image width
            
            assignment = np.arange(n_pixels, dtype=np.int32)  # Start with identity
            
            COLOR_SCALE = 255.0 ** 2
            
            # For each target position i, we have source pixel assignment[i]
            # Color difference: target[i] vs source[assignment[i]]
            color = np.sum((target_flat - source_flat[assignment])**2, axis=1) * COLOR_SCALE
            
            # Spatial difference: where is source pixel assignment[i] originally located vs target position i?
            target_y, target_x = np.divmod(np.arange(n_pixels), w)
            source_original_y, source_original_x = np.divmod(assignment, w)
            spatial = (target_x - source_original_x)**2 + (target_y - source_original_y)**2
            
            # Combined cost 
            heuristics = color * weights_flat + (spatial * proximity_importance)**2
            

            # Main greedy swap algorithm loop
            generation = 0
            current_max_dist = max_distance
            start_time = time.time()
            if self.track_memory:
                # --- START OF MONITORING LOGIC ---
                stop_monitor_event = multiprocessing.Event()
                parent_pid = os.getpid()
                
                monitor_process = multiprocessing.Process(
                    target=display_resource_monitor,
                    kwargs={
                        'parent_pid': parent_pid,
                        'stop_event': stop_monitor_event,
                        'show_spinner': True,
                        'message': "Solving with greedy swap..."
                    }
                )
                
                monitor_process.start()
                
                status = -1 
                try:
                    assignment = self.greedy_loop(swaps_per_generation, max_generations, proximity_importance, w, n_pixels, source_flat, target_flat, weights_flat, assignment, COLOR_SCALE, heuristics, generation, current_max_dist)
                    end_time = time.time()

                finally:
                    # This block ensures the monitor ALWAYS stops, even if smcf.solve() errors out
                    stop_monitor_event.set()
                    monitor_process.join(timeout=2) # Wait up to 2s for it to finish cleanly
                    if monitor_process.is_alive():
                        monitor_process.terminate() # Forcefully stop if it's stuck
                # --- END OF MONITORING LOGIC ---
            else:
                with Progress(
                    SpinnerColumn("dots"),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console,
                    transient=True
                ) as progress:
                    task = progress.add_task(
                        "[#FF8C00]Solving with greedy swap..."
                    )
                    
                    assignment = self.greedy_loop(swaps_per_generation, max_generations, proximity_importance, w, n_pixels, source_flat, target_flat, weights_flat, assignment, COLOR_SCALE, heuristics, generation, current_max_dist)
                    end_time = time.time()
        
                    
            total_time = end_time - start_time
            console.print(f"[green]\u2713[/green] Solution converged in [bold]{total_time:.2f}s[/bold]", highlight=False)

            result = self.apply_transformation(source, assignment)
            
            return assignment, result

    def greedy_loop(self, swaps_per_generation, max_generations, proximity_importance, w, n_pixels, source_flat, target_flat, weights_flat, assignment, COLOR_SCALE, heuristics, generation, current_max_dist):
        while generation < max_generations and current_max_dist >= 2:
            swaps_made = 0
                    
            batch_size = swaps_per_generation
                    
                    # Pick random target positions A and B to swap their source assignments
            apos = np.random.randint(0, n_pixels, size=batch_size)
            ax = apos % w
            ay = apos // w
                    
                    # Pick random nearby positions
            dx = np.random.randint(-int(current_max_dist), int(current_max_dist) + 1, size=batch_size)
            dy = np.random.randint(-int(current_max_dist), int(current_max_dist) + 1, size=batch_size)
                    
            bx = np.clip(ax + dx, 0, w - 1)
            by = np.clip(ay + dy, 0, w - 1)
            bpos = by * w + bx
                    
                    # Filter out self-swaps
            valid_mask = (apos != bpos)
            apos = apos[valid_mask]
            bpos = bpos[valid_mask]
            ax = ax[valid_mask]
            ay = ay[valid_mask]
            bx = bx[valid_mask]
            by = by[valid_mask]
                    
            if len(apos) == 0:
                generation += 1
                continue
                    
                    # Current assignments and costs
            h_a = heuristics[apos]
            h_b = heuristics[bpos]
                    
            source_a = assignment[apos]  # Source pixel currently at target position A
            source_b = assignment[bpos]  # Source pixel currently at target position B
                    
                    # After swap:
                    # - Target position A would get source_b
                    # - Target position B would get source_a
                    
                    # Cost of source_b at target position A
            color_ba = np.sum((target_flat[apos] - source_flat[source_b])**2, axis=1) * COLOR_SCALE
            source_b_original_x = source_b % w
            source_b_original_y = source_b // w
            spatial_ba = (ax - source_b_original_x)**2 + (ay - source_b_original_y)**2
            h_ba = color_ba * weights_flat[apos] + (spatial_ba * proximity_importance)**2
                    
                    # Cost of source_a at target position B
            color_ab = np.sum((target_flat[bpos] - source_flat[source_a])**2, axis=1) * COLOR_SCALE
            source_a_original_x = source_a % w
            source_a_original_y = source_a // w
            spatial_ab = (bx - source_a_original_x)**2 + (by - source_a_original_y)**2
            h_ab = color_ab * weights_flat[bpos] + (spatial_ab * proximity_importance)**2
                    
                    # Check which swaps improve total cost
            improvement = (h_a + h_b) - (h_ba + h_ab)
            beneficial_mask = improvement > 0
                    
            beneficial_apos = apos[beneficial_mask]
            beneficial_bpos = bpos[beneficial_mask]
            beneficial_h_ba = h_ba[beneficial_mask]
            beneficial_h_ab = h_ab[beneficial_mask]
                    
                    # Track which positions have been involved in a swap this batch
            swapped_positions = set()
            swaps_made = 0
                    
            for i in range(len(beneficial_apos)):
                a = beneficial_apos[i]
                b = beneficial_bpos[i]
                        
                        # Skip if either position already swapped this batch
                if a in swapped_positions or b in swapped_positions:
                    continue
                        
                        # Perform swap
                assignment[a], assignment[b] = assignment[b], assignment[a]
                heuristics[a] = beneficial_h_ba[i]
                heuristics[b] = beneficial_h_ab[i]
                        
                swapped_positions.add(a)
                swapped_positions.add(b)
                swaps_made += 1
                    
            generation += 1
                    
                    # Shrink neighborhood
            current_max_dist *= 0.99
            current_max_dist = max(current_max_dist, 2.0)
                    
                    
                    # Stop if converged
            if swaps_made < 10:
                break
        return assignment

    def save_result(self, image: np.ndarray, output_path: str):
        """Save the transformed image."""
        img = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(output_path)


    def compute_cost_matrix_with_spatial(
        self,
        source_flat: np.ndarray,
        target_flat: np.ndarray,
        weights_flat: np.ndarray,
        width: int,
        proximity_importance: int = 10
    ) -> np.ndarray:
        """
        Compute cost matrix with both color and spatial components.
        """
        n_pixels = len(source_flat)
        
        # Color distance
        color_diff = target_flat[:, np.newaxis, :] - source_flat[np.newaxis, :, :]
        color_dist = np.sum(color_diff ** 2, axis=2)
        
        color_dist = color_dist * (255 ** 2)
        
        # Spatial distance
        target_indices = np.arange(n_pixels)
        target_x = target_indices % width
        target_y = target_indices // width
        
        source_indices = np.arange(n_pixels)
        source_x = source_indices % width
        source_y = source_indices // width
        
        dx = target_x[:, np.newaxis] - source_x[np.newaxis, :]
        dy = target_y[:, np.newaxis] - source_y[np.newaxis, :]
        spatial_dist = dx**2 + dy**2

        # Combined cost
        cost_matrix = (color_dist * weights_flat[:, np.newaxis] + 
                    (spatial_dist * proximity_importance)**2)
        
        return cost_matrix

