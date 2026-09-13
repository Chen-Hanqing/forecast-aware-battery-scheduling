from pathlib import Path

import typer

from .config import load_config
from .data import make_demo_data
from .pipeline import run

app = typer.Typer(no_args_is_help=True)

@app.command("make-demo-data")
def demo(output: Path = typer.Option(..., help="Destination CSV"), days: int = 120):
    """Create a realistic, deterministic demo history."""
    make_demo_data(output, days); typer.echo(f"Wrote {output}")

@app.command("run")
def run_command(config: Path = typer.Option(..., "--config", "-c")):
    """Train/select forecast, generate scenarios, optimize, and optionally backtest."""
    metrics=run(load_config(config)); typer.echo(metrics)

if __name__ == "__main__": app()
