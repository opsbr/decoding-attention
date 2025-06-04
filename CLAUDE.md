# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a transformer learning project that implements basic tokenization and simple neural network training using PyTorch. The main implementation is in `main.py` and demonstrates:

- Tokenization using Hugging Face's transformers library with the Qwen/Qwen3-0.6B model
- Manual weight training for next token prediction
- Basic softmax and multinomial sampling for token generation

## Dependencies

The project uses uv for dependency management. Key dependencies include:
- PyTorch (>=2.7.0) for neural networks
- Transformers (>=4.52.3) for tokenizer
- JAXtyping and beartype for type safety
- torchtune and torchao for additional PyTorch utilities

## Development Commands

Since this project uses uv, run Python scripts with:
```bash
uv run python main.py
```

To install dependencies:
```bash
uv sync
```

## Code Architecture

The codebase is currently a single-file implementation (`main.py`) that demonstrates:

1. **Tokenization**: Uses Hugging Face AutoTokenizer to convert text to token IDs
2. **Manual Training**: Implements a simple weight training function that manually sets weights based on expected next tokens
3. **Model**: Uses a basic linear layer (`nn.Linear`) to represent the transformer's final layer
4. **Generation**: Uses softmax and multinomial sampling for token prediction

The code uses strict typing with JAXtyping for tensor shapes and beartype for runtime type checking.