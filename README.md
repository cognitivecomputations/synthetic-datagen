# Synthetic Data Generation

A collection of scripts for generating synthetic data using AI models.

## Features

- Support for OpenAI and compatible API endpoints
- Configurable model selection
- Retry logic for API resilience
- Concurrent processing capabilities
- Environment-based configuration

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/model-context-protocol.git
cd model-context-protocol
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:
   - Copy `.env.local` to `.env`
   - Update the values in `.env` with your API credentials:

```env
OPENAI_API_KEY=your_api_key
OPENAI_ENDPOINT=your_endpoint
OPENAI_MODEL=your_model
```

## Usage

The main script `gen_artifacts.py` provides functionality for generating artifacts using AI models with specific context protocols.

```python
python gen_artifacts.py
```

## Configuration

The following environment variables are supported:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_ENDPOINT`: The API endpoint URL
- `OPENAI_MODEL`: The model identifier to use

## License

[Add your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
