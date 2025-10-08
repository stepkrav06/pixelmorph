from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich import box
from rich.style import Style
from rich.text import Text
from rich.spinner import Spinner
from rich.console import Group
import time
import psutil
from multiprocessing.synchronize import Event as MultiprocessingEvent

def generate_monitor_table(stats: dict, total_memory_mb: float) -> Table:
    """Creates and returns a new Rich Table with the latest stats."""
    table = Table(title="Live Resource Usage (Solver Running)", box=box.ROUNDED, show_header=True, header_style="bold #FF8C00")
    table.add_column("Metric", style="#FF8C00", no_wrap=True, min_width=20)
    table.add_column("Value", style="green", justify="right", min_width=12)
    
    cpu_val = stats.get('cpu', '...')
    mem_val = stats.get('mem_mb', '...')
    
    cpu_str = f"{cpu_val:.1f}%" if isinstance(cpu_val, float) else cpu_val
    mem_str = f"{mem_val:,.1f} MB" if isinstance(mem_val, float) else mem_val

    completed_val = mem_val if isinstance(mem_val, float) else 0.0

    mem_bar = Progress(
        ThresholdBarColumn(bar_width=24),
        expand=True
    )
    task_id = mem_bar.add_task("impact", total=total_memory_mb)
    mem_bar.update(task_id, completed=completed_val)
    mem_label = Text(
        f"{completed_val / (1024):,.1f} / {total_memory_mb / (1024):,.1f} GB",
        style="default",
        justify="right"
    )
    bar_layout = Table.grid(expand=True)
    bar_layout.add_column(width=24)  
    bar_layout.add_column(width=16) 


    bar_layout.add_row(mem_bar, mem_label)


    table.add_row("CPU %", cpu_str)
    table.add_row("Memory (MB)", mem_str)
    table.add_row("Memory Impact", bar_layout)
    return table
    

def display_resource_monitor(parent_pid: int, stop_event: MultiprocessingEvent, show_spinner: bool = False, message: str = "Solving..."):
    """Monitors the parent process's resource usage in a separate process."""
    console = Console()
    stats = {}
    
    try:
        parent_process = psutil.Process(parent_pid)
        parent_process.cpu_percent(interval=None)
        total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        if show_spinner:
           
            spinner = Progress(
                SpinnerColumn("dots", style="#FF8C00"),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
                transient=True
            )
            spinner.add_task(description=f"[#FF8C00]{message}[/#FF8C00]", total=None)
            display_group = Group(spinner, generate_monitor_table(stats, total_memory_mb))
        else:
            display_group = generate_monitor_table(stats, total_memory_mb)
        with Live(display_group, console=console, refresh_per_second=10, transient=True) as live:
            while not stop_event.is_set(): 
                stats['cpu'] = parent_process.cpu_percent(interval=0.5)
                stats['mem_mb'] = parent_process.memory_info().rss / (1024 * 1024)
                
                if show_spinner:
                    live.update(Group(spinner, generate_monitor_table(stats, total_memory_mb)))
                else:
                    live.update(generate_monitor_table(stats, total_memory_mb))

                

    except (psutil.NoSuchProcess, ProcessLookupError):
        # This is expected if the parent process finishes quickly
        pass
    except Exception:
        # Fails silently to prevent a monitoring crash from stopping the main program
        pass

class ThresholdBarColumn(BarColumn):
    """A BarColumn that changes color based on completion percentage."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.thresholds = [
            (20.0, "green"),  # Up to 20%
            (50.0, "yellow"), # Up to 50%
            (100.0, "red"),   # Up to 100%
        ]

    def render(self, task) -> Text:
        """Renders the bar with a color based on the task's percentage."""
        if task.percentage is None:
             return Text("?", style="dim")

        # Choose the style based on the thresholds
        for threshold, color in self.thresholds:
            if task.percentage <= threshold:
                self.complete_style = Style.parse(color)
                break
        
        # Call the original render method with the updated style
        return super().render(task)
    
def extract_progress_to_process(parent_pid: int, message: str, stop_event: MultiprocessingEvent, show_time_elapsed: bool = False):
    """Show progress spinner for code that blocks GIL, using a separate process."""
    console = Console()
    try:
        parent_process = psutil.Process(parent_pid)
        if show_time_elapsed:
            spinner = Progress(
                SpinnerColumn("dots", style="#FF8C00"),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
                transient=True
            )
            spinner.add_task(description=f"[#FF8C00]{message}[/#FF8C00]", total=None)
        else:
            spinner = Spinner("dots", text=f"[#FF8C00]{message}[/#FF8C00]")

        with Live(spinner, console=console, refresh_per_second=10, transient=True) as live:
            while not stop_event.is_set() and parent_process.is_running():
                time.sleep(0.1)
    except (psutil.NoSuchProcess, Exception):
        pass # Fail silently if parent is gone or another error occurs
    except Exception:
        # Fails silently to prevent a monitoring crash from stopping the main program
        pass