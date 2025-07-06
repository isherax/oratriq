"""Main CLI entry point for Oratriq application."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from config.settings import settings
from src.utils.logger import setup_logger

# Initialize console and logger
console = Console()
logger = setup_logger("oratriq.main")


@click.group()
@click.version_option(version="0.1.0", prog_name="Oratriq")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def cli(debug: bool) -> None:
    """Oratriq - AI Presentation Skills Coach.
    
    An AI-powered application that analyzes your presentation skills
    using microphone input and provides personalized recommendations.
    """
    if debug:
        settings.debug = True
        logger.setLevel("DEBUG")
        console.print("[yellow]Debug mode enabled[/yellow]")


@cli.command()
@click.option(
    "--duration",
    "-d",
    default=60,
    help="Recording duration in seconds (default: 60)"
)
@click.option(
    "--output",
    "-o",
    help="Output file path for the recording"
)
def start(duration: int, output: str) -> None:
    """Start a new presentation recording session."""
    console.print(Panel.fit(
        "[bold blue]🎤 Starting Presentation Recording Session[/bold blue]\n"
        f"Duration: {duration} seconds\n"
        "Press Ctrl+C to stop recording early",
        border_style="blue"
    ))
    
    try:
        # TODO: Implement audio recording functionality
        console.print("[green]✓ Recording session started[/green]")
        console.print("[yellow]⚠ Audio recording not yet implemented[/yellow]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Recording stopped by user[/yellow]")
    except Exception as e:
        logger.error(f"Error during recording: {e}")
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    help="Output file path for the analysis report"
)
def analyze(file_path: str, output: str) -> None:
    """Analyze a recorded audio file and generate recommendations."""
    console.print(Panel.fit(
        "[bold green]🔍 Analyzing Presentation Recording[/bold green]\n"
        f"File: {file_path}",
        border_style="green"
    ))
    
    try:
        # TODO: Implement audio analysis functionality
        console.print("[green]✓ Analysis started[/green]")
        console.print("[yellow]⚠ Audio analysis not yet implemented[/yellow]")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
def status() -> None:
    """Show application status and configuration."""
    console.print(Panel.fit(
        "[bold cyan]📊 Application Status[/bold cyan]\n"
        f"Version: 0.1.0\n"
        f"Debug Mode: {settings.debug}\n"
        f"Log Level: {settings.log_level}\n"
        f"STT Provider: {settings.stt_provider}\n"
        f"Audio Format: {settings.audio_format}\n"
        f"Sample Rate: {settings.audio_sample_rate}Hz",
        border_style="cyan"
    ))


@cli.command()
def setup() -> None:
    """Run initial setup and configuration."""
    console.print(Panel.fit(
        "[bold magenta]⚙️ Application Setup[/bold magenta]\n"
        "This will guide you through the initial setup process.",
        border_style="magenta"
    ))
    
    try:
        # Check if directories exist
        directories = [
            settings.recordings_dir,
            settings.transcripts_dir,
            settings.cache_dir,
        ]
        
        for directory in directories:
            if directory.exists():
                console.print(f"[green]✓ {directory} exists[/green]")
            else:
                directory.mkdir(parents=True, exist_ok=True)
                console.print(f"[green]✓ Created {directory}[/green]")
        
        # Check environment variables
        if settings.openai_api_key:
            console.print("[green]✓ OpenAI API key configured[/green]")
        else:
            console.print("[yellow]⚠ OpenAI API key not configured[/yellow]")
            console.print("Set OPENAI_API_KEY environment variable for AI features")
        
        console.print("\n[bold green]Setup completed successfully![/bold green]")
        
    except Exception as e:
        logger.error(f"Error during setup: {e}")
        console.print(f"[red]Error during setup: {e}[/red]")


if __name__ == "__main__":
    cli() 
