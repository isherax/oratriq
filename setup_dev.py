#!/usr/bin/env python3
"""Development setup script for Oratriq."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Oratriq development environment...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create virtual environment
    if not Path("venv").exists():
        if not run_command("python -m venv venv", "Creating virtual environment"):
            sys.exit(1)
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation command based on OS
    if sys.platform == "win32":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install dependencies
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        sys.exit(1)
    
    # Install pre-commit hooks
    if not run_command(f"{pip_cmd} install pre-commit", "Installing pre-commit"):
        print("⚠️ Pre-commit installation failed, continuing...")
    
    if Path(".git").exists():
        if not run_command("pre-commit install", "Installing pre-commit hooks"):
            print("⚠️ Pre-commit hooks installation failed, continuing...")
    
    # Create .env file if it doesn't exist
    if not Path(".env").exists():
        if Path("env.example").exists():
            run_command("cp env.example .env", "Creating .env file from template")
            print("📝 Please edit .env file with your configuration")
        else:
            print("⚠️ env.example not found, please create .env file manually")
    
    # Run application setup
    print("\n🔧 Running application setup...")
    if not run_command(f"{pip_cmd} install -e .", "Installing application in development mode"):
        print("⚠️ Development installation failed, continuing...")
    
    print("\n" + "=" * 50)
    print("🎉 Development environment setup completed!")
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Configure your .env file with API keys")
    print("3. Run: python src/main.py setup")
    print("4. Run: python src/main.py --help")
    print("\nHappy coding! 🚀")


if __name__ == "__main__":
    main() 
