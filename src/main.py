"""Main CLI entry point for Oratriq application."""

import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import time

from config.settings import settings
from src.audio.recorder import AudioRecorder
from src.speech.stt import SpeechToText
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
        # Initialize audio recorder
        recorder = AudioRecorder(
            sample_rate=settings.audio_sample_rate,
            channels=settings.audio_channels,
            chunk_size=settings.audio_chunk_size
        )
        
        # Generate output filename if not provided
        if not output:
            timestamp = int(time.time())
            output = settings.recordings_dir / f"recording_{timestamp}.wav"
        
        # Record audio
        console.print("[green]✓ Recording started...[/green]")
        audio_data = recorder.record_for_duration(duration)
        
        # Save audio file
        recorder.save_audio(audio_data, str(output))
        console.print(f"[green]✓ Recording saved to: {output}[/green]")
        
        # Cleanup
        recorder.cleanup()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Recording stopped by user[/yellow]")
    except Exception as e:
        logger.error(f"Error during recording: {e}")
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option(
    "--duration",
    "-d",
    default=10,
    help="Recording duration in seconds (default: 10)"
)
@click.option(
    "--language",
    "-l",
    default="en-US",
    help="Language code for speech recognition (default: en-US)"
)
@click.option(
    "--save-audio",
    "-s",
    is_flag=True,
    help="Save the audio recording to file"
)
def stt(duration: int, language: str, save_audio: bool) -> None:
    """Convert speech from microphone to text."""
    console.print(Panel.fit(
        "[bold green]🎤 Speech-to-Text Conversion[/bold green]\n"
        f"Duration: {duration} seconds\n"
        f"Language: {language}\n"
        "Press Ctrl+C to stop recording early",
        border_style="green"
    ))
    
    try:
        # Initialize audio recorder
        recorder = AudioRecorder(
            sample_rate=settings.audio_sample_rate,
            channels=settings.audio_channels,
            chunk_size=settings.audio_chunk_size
        )
        
        # Initialize speech-to-text converter
        stt_converter = SpeechToText()
        
        # Record audio
        console.print("[green]✓ Recording started...[/green]")
        console.print("[yellow]Speak now...[/yellow]")
        
        audio_data = recorder.record_for_duration(duration)
        
        # Save audio if requested
        if save_audio:
            timestamp = int(time.time())
            audio_file = settings.recordings_dir / f"stt_recording_{timestamp}.wav"
            recorder.save_audio(audio_data, str(audio_file))
            console.print(f"[green]✓ Audio saved to: {audio_file}[/green]")
        
        # Convert speech to text
        console.print("[green]✓ Converting speech to text...[/green]")
        text = stt_converter.convert_audio_data(audio_data, language)
        
        # Display results
        console.print()
        console.print(Panel.fit(
            f"[bold cyan]Transcription Result:[/bold cyan]\n\n{text}",
            border_style="cyan"
        ))
        
        # Cleanup
        recorder.cleanup()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Recording stopped by user[/yellow]")
    except Exception as e:
        logger.error(f"Error during speech-to-text conversion: {e}")
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
