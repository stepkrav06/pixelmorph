"""
Image Transform + Animation Pipeline
Links image_transform_mcf.py with voronoi_morph_fix.py to create animated morphs.
"""

import argparse
import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Rich imports
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich.markup import escape
from rich import box

console = Console()

# Import the transformer and simulation classes
from .image_transform import ImageTransformer
from .voronoi_morph import VoronoiSimulation


def main():
    parser = argparse.ArgumentParser(
        description='Image Transform + Animation - Transform and animate image morphs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform only (no animation)
  python image_transform_animate.py input.jpg --target target.png --method sparse_mcf --size 128
  
  # Transform and animate
  python image_transform_animate.py input.jpg --target target.png --method sparse_mcf --size 128 --animate --gif-output morph.gif
  
  # Full control over animation parameters
  python image_transform_animate.py input.jpg --target target.png --method greedy_swap --size 256 --animate --gif-output result.gif --total-frames 200 --gif-fps 10
  
  # Animate from pre-computed assignments (skips computation)
  python image_transform_animate.py input.jpg --assignments assignments.json --size 128 --animate --gif-output morph.gif
  
  # Custom output folder
  python image_transform_animate.py input.jpg --target target.png --method sparse_mcf --size 128 --animate --output-dir my_results
        """
    )
    
    # Arguments from image_transform_mcf.py
    parser.add_argument('source', help='Source image path')
    parser.add_argument('--target', default='target.png', help='Target image (default: target.png)')
    parser.add_argument('--weights', default=None, help='Weights image path')
    parser.add_argument('--output', default='output.png', help='Output image path')
    parser.add_argument('--size', type=int, default=128, help='Image size (default: 128)')
    parser.add_argument('--method', 
                       choices=['hungarian', 'mincostflow', 'sparse_mcf', 'greedy_swap'], 
                       default='greedy_swap', 
                       help='''Optimization method (default: greedy_swap)
  hungarian     - Hungarian algorithm (optimal, slow, moderate memory)
  mincostflow   - Min-cost flow (optimal, moderate speed, very high memory)
  sparse_mcf    - Sparse min-cost flow (provably near-optimal (~99%% accuracy), fast, moderate memory)
  greedy_swap   - Greedy swap local search (fast, low memory, good results, recommended for most cases)''')
    parser.add_argument('--k-neighbors', type=int, default=500, 
                       help='Number of neighbors for sparse_mcf (default: 500)')
    parser.add_argument('--max-distance', type=float, default=None,
                       help='Initial max distance for greedy_swap algorithm (default: image width)')
    parser.add_argument('--swaps-per-gen', type=int, default=None,
                       help='Swaps per generation for greedy_swap algorithm (default: auto-tuned based on image size)')
    parser.add_argument('--append-method', action='store_true',
                       help='Append method name to output filename')
    parser.add_argument('--track-memory', action='store_true',
                       help='Track and print memory usage statistics (useful to see for mcf and hungarian)')
    
    # New argument to skip computation and use pre-computed assignments
    parser.add_argument('--assignments', default=None, 
                       help='Path to pre-computed assignments JSON file (skips computation step)')
    
    # Animation-specific arguments
    parser.add_argument('--animate', action='store_true',
                       help='Generate animated GIF of the transformation')
    parser.add_argument('--gif-output', default='morph_animation.gif',
                       help='Output path for animated GIF (default: morph_animation.gif)')
    parser.add_argument('--total-frames', type=int, default=140,
                       help='Total frames in output GIF (default: 140)')
    parser.add_argument('--gif-fps', type=int, default=8,
                       help='Frame rate of output GIF (default: 8)')
    parser.add_argument('--output-res', type=int, default=None,
                       help='Output resolution for animation (default: 2.5x input size)')
    
    # Output directory argument
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for all files (default: auto-generated timestamp folder)')
    parser.add_argument('--no-timestamp', action='store_true',
                   help='Omit timestamp from auto-generated folder names')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.assignments and not args.animate:
        print("Warning: --assignments provided but --animate not specified.")
        args.animate = True
    
    if not args.assignments and not args.target:
        parser.error("Either --target or --assignments must be provided")
    
    # Determine output directory path but don't create it yet
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Auto-generate folder name with timestamp and method
        timestamp = "" if args.no_timestamp else f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        source_name = Path(args.source).stem
        method_name = "precomputed" if args.assignments else args.method
        folder_name = f"{source_name}_{method_name}{timestamp}"
        output_dir = Path("outputs") / folder_name
        if output_dir.exists() and args.no_timestamp:
            counter = 1
            while output_dir.exists():
                folder_name = f"{source_name}_{method_name}_{counter}"
                output_dir = Path("outputs") / folder_name
                counter += 1
    
    # Handle pre-computed assignments case
    if args.assignments:
        console.rule("[bold #FF8C00]Validating inputs", style="#FF8C00")
        console.print()
        
        assignments_path = Path(args.assignments)
        if not assignments_path.exists():
            console.print(f"[red]Error:[/red] Assignments file not found: {assignments_path}")
            sys.exit(1)
        
        # Load and validate assignments
        try:
            with open(assignments_path, 'r') as f:
                assignments_data = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            console.print(f"[red]Error:[/red] Failed to load assignments file: {e}")
            sys.exit(1)
        
        # Validate source image exists
        source_path = Path(args.source)
        if not source_path.exists():
            console.print(f"[red]Error:[/red] Source image not found: {source_path}")
            sys.exit(1)
        
        # If size not provided, infer from assignments
        if args.size == 128:  # default value
            inferred_size = int(len(assignments_data) ** 0.5)
            if inferred_size * inferred_size == len(assignments_data):
                args.size = inferred_size
                console.print(f"[green]\u2713[/green] Inferred grid size: [bold]{args.size}x{args.size}[/bold]")
            else:
                console.print(f"[yellow]Warning:[/yellow] Could not infer grid size from assignments, using default: {args.size}")
        
        # Now create output directory after validation
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy source and assignments to output directory
        source_copy = output_dir / f"source{source_path.suffix}"
        shutil.copy2(source_path, source_copy)
        
        assignments_copy = output_dir / "assignments.json"
        shutil.copy2(assignments_path, assignments_copy)
        
        transform_json_path = assignments_copy
        output_path = None  # No transformed image when using pre-computed assignments
        
    else:
        # Original workflow - compute assignments
        # First validate that all required files exist
        source_path = Path(args.source)
        if not source_path.exists():
            console.print(f"[red]Error:[/red] Source image not found: {source_path}")
            sys.exit(1)
        
        target_path = Path(args.target)
        if not target_path.exists():
            console.print(f"[red]Error:[/red] Target image not found: {target_path}")
            sys.exit(1)
        
        # Validate weights if provided
        if args.weights:
            weights_path = Path(args.weights)
            if not weights_path.exists():
                console.print(f"[red]Error:[/red] Weights image not found: {weights_path}")
                sys.exit(1)
    
        # Step 1: Run transformation with configuration table
        console.print()
        console.rule("[bold #FF8C00]Computing transformation", style="#FF8C00")
        console.print()
        
        # Configuration summary table
        config_table = Table(title="Configuration", box=box.ROUNDED, show_header=True, header_style="bold #FF8C00")
        config_table.add_column("Parameter", style="#FF8C00", no_wrap=True, min_width=20)
        config_table.add_column("Value", style="white", min_width=12)

        config_table.add_row("Method", f"[bold]{args.method}[/bold]")
        config_table.add_row("Image Size", f"[bold]{args.size}×{args.size}[/bold]")
        if args.method == 'sparse_mcf':
            config_table.add_row("K-neighbors", f"[bold]{args.k_neighbors}[/bold]")
        elif args.method == 'greedy_swap':
            if args.max_distance:
                config_table.add_row("Max Distance", f"[bold]{args.max_distance}[/bold]")
            if args.swaps_per_gen:
                config_table.add_row("Swaps/Gen", f"[bold]{args.swaps_per_gen:,}[/bold]")
        config_table.add_row("Output Directory", str(output_dir))
        if args.animate:
            config_table.add_row("Animation", "[green]Enabled[/green]")
            config_table.add_row("Total Frames", f"[bold]{args.total_frames}[/bold]")
            config_table.add_row("GIF FPS", f"[bold]{args.gif_fps}[/bold]")
        else:
            config_table.add_row("Animation", "[dim]Disabled[/dim]")
        if args.track_memory:
            config_table.add_row("Resource Tracking", "[green]Enabled[/green]")
        else:
            config_table.add_row("Resource Tracking", "[dim]Disabled[/dim]")
        
        console.print(config_table)
        console.print()
        
        transformer = ImageTransformer(args.target, args.weights, args.size, track_memory=args.track_memory)
        
        if args.method == 'hungarian':
            assignment, result = transformer.transform_hungarian(args.source)
        elif args.method == 'mincostflow':
            assignment, result = transformer.transform_mincostflow(args.source)
        elif args.method == 'sparse_mcf':
            assignment, result = transformer.transform_sparse_mcf(args.source, args.k_neighbors)
        elif args.method == 'greedy_swap':
            assignment, result = transformer.transform_greedy_swap(args.source, args.max_distance, args.swaps_per_gen)
        
        # Transformation successful - now create output directory and save files
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy source, target, and weights to output directory
        source_copy = output_dir / f"source{source_path.suffix}"
        shutil.copy2(source_path, source_copy)
        
        target_copy = output_dir / f"target{target_path.suffix}"
        shutil.copy2(target_path, target_copy)
        
        weights_copy = None
        if args.weights:
            weights_copy = output_dir / f"weights{weights_path.suffix}"
            shutil.copy2(weights_path, weights_copy)
        
        
        # Save transformed image to output directory
        output_filename = Path(args.output).name
        if args.append_method:
            base, ext = os.path.splitext(output_filename)
            output_filename = f"{base}_{args.method}{ext}"
        
        output_path = output_dir / output_filename
        transformer.save_result(result, str(output_path))
        
        # Save transformation JSON to output directory
        transform_json_path = output_path.with_suffix('').with_suffix('.json').with_name(output_path.stem + '_transform.json')
        with open(transform_json_path, 'w') as json_file:
            json.dump(assignment.tolist(), json_file)
    
    # Step 2: Animate if requested
    if args.animate:
        console.print()
        console.rule("[bold #FF8C00]Generating animation", style="#FF8C00")
        console.print()
        
        # Create simulation
        if args.output_res is None:
            args.output_res = round(args.size * 2.5)  # Default to 2.5x input size
        sim = VoronoiSimulation(
            source_image_path=args.source,
            assignments_path=str(transform_json_path),
            output_res=args.output_res,
            grid_size=args.size
        )
        
        # Generate animation in output directory
        gif_filename = Path(args.gif_output).name
        if args.append_method and not args.assignments:
            base, ext = os.path.splitext(gif_filename)
            gif_filename = f"{base}_{args.method}{ext}"
        
        gif_output = output_dir / gif_filename
        
        sim.animate(
            total_frames=args.total_frames,
            gif_fps=args.gif_fps,
            output_path=str(gif_output)
        )
        
        console.print()
        console.print("[green bold]\u2713 Animation complete![/green bold]")
        console.print()
        
        # Create file tree for output summary
        tree = Tree(f"[bold #FF8C00]{output_dir.absolute()}[/bold #FF8C00]")

        source_copy_path = output_dir / f"source{Path(args.source).suffix}"
        tree.add(f"source{Path(args.source).suffix} [dim]({format_file_size(source_copy_path)})[/dim]")

        if args.assignments:
            assignments_copy_path = output_dir / "assignments.json"
            tree.add(f"assignments.json [dim]({format_file_size(assignments_copy_path)})[/dim]")
        else:
            target_copy_path = output_dir / f"target{Path(args.target).suffix}"
            tree.add(f"target{Path(args.target).suffix} [dim]({format_file_size(target_copy_path)})[/dim]")
            if args.weights:
                weights_copy_path = output_dir / f"weights{Path(args.weights).suffix}"
                tree.add(f"weights{Path(args.weights).suffix} [dim]({format_file_size(weights_copy_path)})[/dim]")
            if output_path:
                tree.add(f"{output_path.name} [dim]({format_file_size(output_path)})[/dim]")
            tree.add(f"{transform_json_path.name} [dim]({format_file_size(transform_json_path)})[/dim]")

        tree.add(f"[bold]{gif_output.name}[/bold] [dim]({format_file_size(gif_output)})[/dim]")
        
        panel = Panel(
            tree,
            title="[bold #FF8C00]All outputs saved[/bold #FF8C00]",
            border_style="#FF8C00",
            box=box.ROUNDED
        )
        console.print(panel)
    else:
        console.print()
        console.print("[green]\u2713 Transformation complete (no animation requested)[/green]", highlight=False)
        console.print()
        
        # Create file tree for output summary
        tree = Tree(f"[bold #FF8C00]{output_dir.absolute()}[/bold #FF8C00]")
        source_copy_path = output_dir / f"source{Path(args.source).suffix}"
        tree.add(f"source{Path(args.source).suffix} [dim]({format_file_size(source_copy_path)})[/dim]")

        if not args.assignments:
            target_copy_path = output_dir / f"target{Path(args.target).suffix}"
            tree.add(f"target{Path(args.target).suffix} [dim]({format_file_size(target_copy_path)})[/dim]")
            if args.weights:
                weights_copy_path = output_dir / f"weights{Path(args.weights).suffix}"
                tree.add(f"weights{Path(args.weights).suffix} [dim]({format_file_size(weights_copy_path)})[/dim]")
            if output_path:
                tree.add(f"[bold]{output_path.name}[/bold] [dim]({format_file_size(output_path)})[/dim]")
            tree.add(f"{transform_json_path.name} [dim]({format_file_size(transform_json_path)})[/dim]")
        
        panel = Panel(
            tree,
            title="[bold #FF8C00]Outputs saved[/bold #FF8C00]",
            border_style="#FF8C00",
            box=box.ROUNDED
        )
        console.print(panel)
        
        console.print()
        console.print("[dim]To animate this transformation, run:[/dim]")
        console.print(f"[dim]  python {sys.argv[0]} {args.source} --assignments {transform_json_path} --size {args.size} --animate[/dim]", highlight=False)

def format_file_size(path):
    """Format file size in human-readable format"""
    size = path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def format_file_with_link(path, display_name=None):
    """Format file with size and clickable link"""
    if display_name is None:
        display_name = path.name
    
    size = format_file_size(path)
    # file:// URL for local files
    file_url = path.absolute().as_uri()
    
    return f"[link={file_url}]{escape(display_name)}[/link] [dim]({size})[/dim]"

if __name__ == '__main__':
    main()