# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

Run the Streamlit app:
```bash
uv run streamlit run main.py --server.runOnSave=true
```

Install dependencies:
```bash
uv sync
```

## Architecture Overview

This is an educational Transformer learning repository structured as a Streamlit application that demonstrates Transformer inference step-by-step using the Qwen3-0.6B model.

### Application Structure

The app uses Streamlit's navigation system with multiple pages:

- **main.py**: Entry point with navigation setup and README content
- **chapter1/**: Complete implementation of tokenization and sampling
- **chapter2/**: Placeholder for future attention mechanisms content

### Chapter 1 Architecture

Chapter 1 demonstrates the complete Transformer inference pipeline through 5 interconnected sections:

1. **section1.py**: Tokenizer initialization (BPE) and vocabulary analysis
2. **section2.py**: Transformer inference to generate logits
3. **section3.py**: Probability distribution with sampling filters (temperature, top-k, top-p, min-p)
4. **section4.py**: Token sampling using multinomial distribution with resampling controls
5. **section5.py**: Autoregressive generation with toggle controls and visualization

### Key Dependencies

- **PyTorch**: CPU-only configuration via pytorch-cpu index for educational accessibility
- **Transformers**: Hugging Face library for Qwen3-0.6B model
- **Streamlit**: Interactive web interface with session state management
- **st-annotated-text**: Token visualization with ID annotations
- **Altair**: Primary visualization library for interactive charts and heatmaps

### Session State Management

Critical session state variables:
- `input_text`: Current text being processed (updated during autoregressive generation)
- `show_edit_modal`: Controls text editing modal visibility
- Sampling parameters: `temperature`, `top_k`, `top_p`, `min_p`
- Resampling state: `current_sampled_token`, `force_resample`, `last_input_text`
- Autoregressive control: `autoregressive_started`
- Cross-section coordination: `sampled_token` (shared for highlighting across visualizations)

### Data Flow Architecture

1. **Section 1**: Loads tokenizer, handles pending input text updates, provides text editing UI with vocabulary analysis
2. **Section 2**: Runs Transformer inference to generate logits tensor with interactive Altair heatmap visualization
3. **Section 3**: Applies sampling filters and creates comparative probability distribution charts
4. **Section 4**: Performs multinomial sampling with intelligent resampling controls and cross-section coordination
5. **Section 5**: Manages autoregressive generation with sophisticated state-aware UI and visual flow design

### Important Implementation Details

- **Input Text Synchronization**: Section 1 handles `pending_input_text` updates from autoregressive generation with widget-key state management to prevent timing issues
- **Advanced Logits Visualization**: Section 2 uses Altair heatmaps with per-position normalization, dynamic scaling based on sequence length, and interactive tooltips showing predictions and ranks
- **Intelligent Resampling Logic**: Section 4 tracks input text changes and forces resampling when text changes to prevent stale cached samples, with session state coordination across sections
- **Autoregressive Control**: Section 5 uses sophisticated state-aware UI with dynamic interface adjustments and visual flow design for generation control
- **Robust Unicode Handling**: Token decoding functions handle non-printable characters, variation selectors, and special Unicode with graceful fallbacks
- **Cross-Section Visualization Coordination**: Shared session state enables highlighting of sampled tokens across all visualizations for educational clarity
- **Educational Transparency**: Each section displays its source code using `inspect.getsource()` for learning purposes

### Streamlit Configuration

- Theme: Light base with navy primary color (#000080)
- Layout: Centered for optimal readability
- Logo: attention.png displayed in navigation
