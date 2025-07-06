# Oratriq - AI Presentation Skills Coach

An AI-powered application that uses computer microphone input to convert speech to text and generate recommendations for improving presentation skills.

## Project Overview

This application provides real-time feedback on presentation skills by:
- Capturing audio from the computer's microphone
- Converting speech to text using AI
- Analyzing presentation patterns and delivery
- Generating actionable recommendations for improvement

## Project Structure

```
oratriq/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
├── .env.example             # Environment variables template
├── .cursor/                 # Cursor IDE configuration
│   └── rules/
│       └── project-rules.mdc # Project coding standards and guidelines
├── config/
│   ├── __init__.py
│   └── settings.py          # Application configuration
├── src/
│   ├── __init__.py
│   ├── main.py              # Main CLI entry point
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── recorder.py      # Audio recording functionality
│   │   └── processor.py     # Audio processing utilities
│   ├── speech/
│   │   ├── __init__.py
│   │   ├── stt.py          # Speech-to-text conversion
│   │   └── analyzer.py     # Speech analysis
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── recommender.py  # AI recommendation engine
│   │   └── models.py       # AI model configurations
│   └── utils/
│       ├── __init__.py
│       ├── logger.py       # Logging utilities
│       └── helpers.py      # General helper functions
├── tests/
│   ├── __init__.py
│   ├── test_audio.py
│   ├── test_speech.py
│   └── test_ai.py
└── data/
    ├── recordings/         # Audio recordings (gitignored)
    └── transcripts/        # Speech transcripts (gitignored)
```

## Features

### Phase 1 (MVP - Command Line Interface)
- [ ] Real-time microphone audio capture
- [ ] Speech-to-text conversion
- [ ] Basic presentation analysis
- [ ] Command-line recommendations output

### Phase 2 (Enhanced Features)
- [ ] Advanced speech pattern analysis
- [ ] Confidence scoring
- [ ] Historical performance tracking
- [ ] Export reports

### Phase 3 (Advanced Features)
- [ ] Real-time feedback during presentations
- [ ] Integration with presentation software
- [ ] Web interface
- [ ] Multi-language support

## Technology Stack

- **Python 3.8+**: Core application language
- **Speech Recognition**: Audio processing and STT
- **OpenAI API**: Advanced text analysis and recommendations
- **PyAudio**: Audio recording and playback
- **Click**: Command-line interface framework
- **Pydantic**: Data validation and settings management

## Development Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Maximum line length: 88 characters (Black formatter)
- Use meaningful variable and function names

### Testing
- Write unit tests for all modules
- Maintain minimum 80% code coverage
- Use pytest for testing framework
- Mock external dependencies in tests

### Documentation
- Use docstrings for all public functions and classes
- Follow Google docstring format
- Keep README updated with new features
- Document API endpoints and configuration options

### Error Handling
- Implement comprehensive error handling
- Use custom exception classes for domain-specific errors
- Log errors with appropriate levels
- Provide user-friendly error messages

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Microphone access
- OpenAI API key (for advanced features)

### Installation
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and configure your settings
6. Run the application: `python src/main.py`

## Usage

### Basic Usage
```bash
# Start a presentation session
python src/main.py start

# Analyze a recorded audio file
python src/main.py analyze --file path/to/recording.wav

# Get help
python src/main.py --help
```

### Configuration
The application can be configured through:
- Environment variables (see `.env.example`)
- Configuration file (`config/settings.py`)
- Command-line arguments

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the coding guidelines
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions, please open an issue on the GitHub repository. 
