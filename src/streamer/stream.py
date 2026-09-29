import questionary
from rich.console import Console
import sys
import os

from stream_ import stream_history, LiveEventStreamPlay

console = Console()

os.system('cls' if os.name=="nt" else "clear")
console.print("Welcome to Customer360!!", style="bold cyan")
console.print('\n')

scenario = questionary.select(
    "",
    choices=[
        questionary.Choice("Scenario 1", value="scenario_01"),
        questionary.Choice("Scenario 2", value="scenario_02"),
        questionary.Choice("Scenario 3", value="scenario_03")
    ],
    style=questionary.Style([
        ('qmark', 'fg:cyan bold'),
        ('question', 'bold'),
        ('answer', 'fg:green bold'),
        ('pointer', 'fg:cyan bold'),
        ('highlighted', 'fg:cyan bold'),
    ])
).ask()

if scenario is None:
    console.print("[red]Operation cancelled by User[/red]")
    sys.exit(0)

console.print(f"Initializing [green]{scenario}[/green] sequence.....\n")
console.print("[yellow]Loading customer history...[/yellow]\n")

try:
    with console.status("[bold green] Customer History...") as status:
        stream_history(scenario)
except Exception as err:
    console.print(f"[red] History Streaming failed : [/red] { err }")
    sys.exit(1)

console.print("[bold green]History seed reading completed....[/bold green]\n")

console.print("[yellow]Streaming...[/yellow]\n")

streamer = LiveEventStreamPlay(scenario, publish_event= lambda event: print(f"[EVENT] {event['event_id']} | {event['ingestion_time']} | {event['event_time']}"))
streamer.replay()