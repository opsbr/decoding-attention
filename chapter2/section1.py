import cache
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    LogitsProcessorList,
)
import inspect
import pandas as pd  # type: ignore
import altair as alt


def section1(model_name):
    pretrained = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(text):
        return tokenizer(text, return_tensors="pt").input_ids.squeeze(0)

    def decode(tokens):
        return tokenizer.decode(tokens, skip_special_tokens=True)

    def sample(logits):
        logits = logits[-1].unsqueeze(0)  # Use the last logits as a batch
        warpers = [
            TemperatureLogitsWarper(0.7),
            TopKLogitsWarper(20),
            TopPLogitsWarper(0.8),
        ]
        x = LogitsProcessorList(warpers)(torch.LongTensor(), logits)
        x = torch.nn.functional.softmax(x.squeeze(0), dim=-1)
        return torch.multinomial(x, num_samples=1)

    return pretrained, tokenize, decode, sample


def run(model_name):
    st.markdown(
        """
### Section 1: Recap Chapter 1

Before talking about embedding, let's recap what we learned in the previous chapter. Here is the compact version of the functions that we used in Chapter 1:
"""
    )
    st.code(inspect.getsource(section1), line_numbers=True)

    pretrained, tokenize, decode, sample = section1(model_name)

    st.markdown(
        """
#### Step 1: Tokenize input
The first step of the decoding process is to tokenize the input text. The `tokenize()` function returns a tensor of token IDs like below:
"""
    )
    tokens = tokenize("Hello world!")
    st.code('tokens = tokenize("Hello world!")')
    create_tokens_matrix_table(tokens.tolist(), decode)

    st.markdown(
        """
#### Step 2: Infer logits
Next, we pass the tokens to the model to get the logits for the next token prediction. While logits are predicted for all positions as a 2D array, we only need the last position for sampling the next token (highlighted in the matrix below):
"""
    )
    logits = pretrained(tokens.unsqueeze(0)).logits.squeeze(0)
    st.code(
        f"""
       logits = pretrained(tokens.unsqueeze(0)).logits.squeeze(0)  # unsqueeze/squeeze are for batch dimension
       logits.shape # {logits.shape}"""
    )
    create_logits_matrix_table(logits)

    st.markdown(
        """
#### Step 3: Sample next token
Lastly, we sample the next token from the logits using the `sample()` function. The sampled token ID is then concatenated to the original tokens to form a new sequence (called autoregressive generation):
"""
    )
    next_token = sample(logits)
    st.code(
        f"""
        next_token = sample(logits)  # ID = {next_token.item()}
        new_tokens = torch.cat([tokens, next_token])"""
    )
    create_tokens_with_next_matrix_table(tokens.tolist(), next_token.item(), decode)

    # Summary and Progress
    with st.container(border=True):
        st.markdown("**📚 Section Summary**")
        st.markdown(
            "We reviewed the Chapter 1 pipeline: tokenization converts text to token IDs, the model generates logits, and sampling produces the next token for autoregressive generation."
        )

        # Progress indicator
        st.progress(1 / 5, text="Step 1/5: Recap Chapter 1 ✅")

    st.divider()

    return pretrained, tokenize, decode, sample


def create_tokens_matrix_table(tokens, decode):
    # Create a 2-row matrix showing token IDs and decoded tokens
    matrix_data = []

    for i, token_id in enumerate(tokens):
        # Get decoded token using the decode function
        try:
            decoded_token = decode([token_id])
            # Replace newlines and other problematic characters
            decoded_token = decoded_token.replace("\n", "\\n").replace("\t", "\\t")
            if not decoded_token.strip():
                decoded_token = "⎵"  # Space symbol for empty/whitespace
        except:
            decoded_token = "?"

        # Decoded token row (top)
        matrix_data.append(
            {
                "Position": f"Token_{i}",
                "Row": "Text",
                "Value": decoded_token,
                "Order": i,
                "HasBorder": False,
            }
        )

        # Token ID row (bottom)
        matrix_data.append(
            {
                "Position": f"Token_{i}",
                "Row": "IDs",
                "Value": str(token_id),
                "Order": i,
                "HasBorder": True,
            }
        )

    df = pd.DataFrame(matrix_data)

    # Create chart for token IDs only (with borders and background)
    rectangles = (
        alt.Chart(df)
        .transform_filter(alt.datum.HasBorder)
        .mark_rect(stroke="black", strokeWidth=1, fill="lightblue")
        .encode(
            x=alt.X("Position:N", axis=None, sort=alt.SortField("Order")),
            y=alt.Y("Row:N", axis=None),
            tooltip=alt.value(None),
        )
    )

    text = (
        alt.Chart(df)
        .mark_text(align="center", baseline="middle", fontSize=9, color="black")
        .encode(
            x=alt.X("Position:N", axis=None, sort=alt.SortField("Order")),
            y=alt.Y("Row:N", axis=None),
            text=alt.Text("Value:N"),
            tooltip=alt.value(None),
        )
    )

    chart = (rectangles + text).properties(width=160, height=65)
    st.altair_chart(chart, width="content")


def create_tokens_with_next_matrix_table(tokens, next_token, decode):
    # Create a 2-row matrix showing current tokens + next token with different colors
    matrix_data = []

    # Add current tokens
    for i, token_id in enumerate(tokens):
        # Get decoded token using the decode function
        try:
            decoded_token = decode([token_id])
            decoded_token = decoded_token.replace("\n", "\\n").replace("\t", "\\t")
            if not decoded_token.strip():
                decoded_token = "⎵"
        except:
            decoded_token = "?"

        # Decoded token row (top)
        matrix_data.append(
            {
                "Position": f"Token_{i}",
                "Row": "Text",
                "Value": decoded_token,
                "TokenType": "current",
                "Order": i,
                "HasBorder": False,
            }
        )

        # Token ID row (bottom)
        matrix_data.append(
            {
                "Position": f"Token_{i}",
                "Row": "IDs",
                "Value": str(token_id),
                "TokenType": "current",
                "Order": i,
                "HasBorder": True,
            }
        )

    # Add next token
    next_idx = len(tokens)
    try:
        next_decoded = decode([next_token])
        next_decoded = next_decoded.replace("\n", "\\n").replace("\t", "\\t")
        if not next_decoded.strip():
            next_decoded = "⎵"
    except:
        next_decoded = "?"

    # Next decoded token row (top)
    matrix_data.append(
        {
            "Position": f"Token_{next_idx}",
            "Row": "Text",
            "Value": next_decoded,
            "TokenType": "next",
            "Order": next_idx,
            "HasBorder": False,
        }
    )

    # Next token ID row (bottom)
    matrix_data.append(
        {
            "Position": f"Token_{next_idx}",
            "Row": "IDs",
            "Value": str(next_token),
            "TokenType": "next",
            "Order": next_idx,
            "HasBorder": True,
        }
    )

    df = pd.DataFrame(matrix_data)

    # Calculate width to maintain consistent cell sizes
    original_token_count = len(tokens)
    total_token_count = len(tokens) + 1
    width = int(160 * total_token_count / original_token_count)

    # Create separate charts for different styling
    # Chart for current token IDs (blue background)
    current_ids = (
        alt.Chart(df)
        .transform_filter("datum.HasBorder && datum.TokenType == 'current'")
        .mark_rect(stroke="black", strokeWidth=1, fill="lightblue")
        .encode(
            x=alt.X("Position:N", axis=None, sort=alt.SortField("Order")),
            y=alt.Y("Row:N", axis=None),
            tooltip=alt.value(None),
        )
    )

    # Chart for next token IDs (green background)
    next_ids = (
        alt.Chart(df)
        .transform_filter("datum.HasBorder && datum.TokenType == 'next'")
        .mark_rect(stroke="black", strokeWidth=1, fill="lightgreen")
        .encode(
            x=alt.X("Position:N", axis=None, sort=alt.SortField("Order")),
            y=alt.Y("Row:N", axis=None),
            tooltip=alt.value(None),
        )
    )

    # Combine rectangle charts
    rectangles = current_ids + next_ids

    text = (
        alt.Chart(df)
        .mark_text(align="center", baseline="middle", fontSize=9, color="black")
        .encode(
            x=alt.X("Position:N", axis=None, sort=alt.SortField("Order")),
            y=alt.Y("Row:N", axis=None),
            text=alt.Text("Value:N"),
            tooltip=alt.value(None),
        )
    )

    chart = (rectangles + text).properties(width=width, height=65)
    st.altair_chart(chart, width="content")


def create_logits_matrix_table(logits, vocab_size_limit=10):
    # Create a 2D matrix showing logits with limited vocab size
    num_positions = logits.shape[0]
    vocab_size = logits.shape[1]

    # Select top and bottom vocab indices to show with "..." separator
    half_limit = (vocab_size_limit - 1) // 2  # Reserve 1 column for "..."
    top_indices = list(range(half_limit))
    bottom_indices = list(range(vocab_size - half_limit, vocab_size))

    # Create matrix data with proper ordering
    matrix_data = []
    vocab_order = []

    # Build vocabulary order: position labels first, then top indices, then "...", then bottom indices
    vocab_order.append("pos_labels")  # Position labels column at left
    for vocab_idx in top_indices:
        vocab_order.append(f"{vocab_idx:05d}")  # Zero-padded for proper sorting
    vocab_order.append("99999")  # Middle position for "..."
    for vocab_idx in bottom_indices:
        vocab_order.append(f"{vocab_idx:05d}")

    # Unicode subscript mapping for digits 0-9
    subscripts = {
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉",
    }

    for pos in range(num_positions):
        # Add position label as first column
        pos_str = str(pos)
        subscript_pos = "".join(subscripts.get(digit, digit) for digit in pos_str)
        matrix_data.append(
            {
                "Position": pos,
                "Vocab": "pos_labels",
                "Value": f"pos{subscript_pos}",
                "IsLastRow": pos == num_positions - 1,
                "IsPositionLabel": True,
            }
        )

        # Add top indices
        for vocab_idx in top_indices:
            logit_val = logits[pos, vocab_idx].item()
            matrix_data.append(
                {
                    "Position": pos,
                    "Vocab": f"{vocab_idx:05d}",
                    "Value": f"{logit_val:.2f}",
                    "IsLastRow": pos == num_positions - 1,
                    "IsPositionLabel": False,
                }
            )

        # Add "..." separator column
        matrix_data.append(
            {
                "Position": pos,
                "Vocab": "99999",
                "Value": "...",
                "IsLastRow": pos == num_positions - 1,
                "IsPositionLabel": False,
            }
        )

        # Add bottom indices
        for vocab_idx in bottom_indices:
            logit_val = logits[pos, vocab_idx].item()
            matrix_data.append(
                {
                    "Position": pos,
                    "Vocab": f"{vocab_idx:05d}",
                    "Value": f"{logit_val:.2f}",
                    "IsLastRow": pos == num_positions - 1,
                    "IsPositionLabel": False,
                }
            )

    df = pd.DataFrame(matrix_data)

    # Convert Position column to string for proper sorting
    df["Position"] = df["Position"].astype(str)

    # Create separate charts for different color conditions (excluding position labels)
    # Last row (orange)
    last_row_chart = (
        alt.Chart(df)
        .transform_filter("datum.IsLastRow && !datum.IsPositionLabel")
        .mark_rect(stroke="black", strokeWidth=1, fill="orange")
        .encode(
            x=alt.X("Vocab:O", axis=None, sort=vocab_order),
            y=alt.Y("Position:O", axis=None),
            tooltip=alt.value(None),
        )
    )

    # Other rows (light coral)
    other_rows_chart = (
        alt.Chart(df)
        .transform_filter("!datum.IsLastRow && !datum.IsPositionLabel")
        .mark_rect(stroke="black", strokeWidth=1, fill="lightgrey")
        .encode(
            x=alt.X("Vocab:O", axis=None, sort=vocab_order),
            y=alt.Y("Position:O", axis=None),
            tooltip=alt.value(None),
        )
    )

    # Calculate dimensions for square cells
    # We have: pos_labels + top_indices + "..." + bottom_indices
    num_vocab_cols = 1 + len(top_indices) + 1 + len(bottom_indices)
    cell_size = 40
    chart_width = num_vocab_cols * cell_size
    chart_height = num_positions * cell_size

    # Combine rectangle charts (no background/border for position labels)
    chart = (last_row_chart + other_rows_chart).properties(
        width=chart_width, height=chart_height
    )

    # Add text overlay
    text_chart = (
        alt.Chart(df)
        .mark_text(align="center", baseline="middle", fontSize=10, color="black")
        .encode(
            x=alt.X("Vocab:O", axis=None, sort=vocab_order),
            y=alt.Y("Position:O", axis=None),
            text=alt.Text("Value:N"),
            tooltip=alt.value(None),
        )
    )

    st.altair_chart(chart + text_chart, width="content")
