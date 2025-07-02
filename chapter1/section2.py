import inspect
import streamlit as st
import torch
from transformers import AutoModelForCausalLM
import pandas as pd  # type: ignore
import altair as alt
import cache
from section1 import decode_token


def section2(model_name, tokens):
    # Initialize the model from Hugging Face's pretrained data.
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Convert tokens to a tensor as a batch of size 1.
    tokens = torch.tensor([tokens])
    # Actual inference happens here. We'll implement this in later chapters.
    output = model(tokens)
    # Output the "logits" at all positions.
    # Here, our batch size is 1, so we take the first element only.
    return output.logits[0]


def run(model_name, tokens, tokenizer):
    st.markdown(
        """
### Section 2: Logits inference with Transformer

In this section, we actually run a Transformer model against your input text. We won't talk about the details of the model yet, but for now you can think of it as just a function (`model()`) that takes a sequence of integers and returns another form of numbers called "logits". While we work on the inside of `model()` from the next chapter, let's understand what they output now. Here is the code that runs the model and gets the logits:
    """
    )
    st.code(inspect.getsource(section2), line_numbers=True)
    st.caption(
        "Note: You notice we use a batch here because the model expects a batch of token sequences. Since we only care about one sequence, we use a batch size of 1."
    )

    logits = section2(model_name, tokens)
    logits = logits.detach()  # Detach to avoid tracking gradients

    st.markdown(
        """
#### What are logits?
As you might know, `model()` i.e. Transformer (or Language model in general) predicts the next token in a sequence. So, the output of `model()` is the sequence of next token predictions at each position in the input sequence. These predictions are often called "logits". Let's see what they look like:
"""
    )

    st.code(
        f"""
logits.shape = {logits.shape}
# For example, logits at the first position:
logits[0] = {logits[0]}"""
    )

    st.markdown(
        """
The logits are 2-dimensional: the first dimension is the sequence position, and the second dimension is the vocabulary size. So, for each position, the model outputs confidence scores for all individual tokens in the vocabulary. In other words, instead of predicting one single next token at each position, the model predicts a distribution of scores over all tokens in the vocabulary (the higher, the more likely to be the next token).

#### Logits visualization

The next heatmap is a visualization of logits. The x-axis is the vocabulary index, and the y-axis is the sequence position. The color intensity represents the logit value for that token at that position. The darker the color, the higher the logit value, meaning that token is more likely to be the next token at that position. You can hover over the heatmap to see the raw logit value of the potential next token and their ranks at each position (Note: Only showing a few top ranks here, but the raw data is for all tokens as you saw above):
"""
    )

    create_logits_heatmap(logits, tokenizer, tokens)

    st.markdown(
        """
When training the model, it is optimized to predict these logits to be higher for the next token in the training data and lower for all other tokens. So, if your input text is similar to patterns the Qwen3 model learned during training, the top-ranked token could be the same as in your input, but most of the time it's not.

#### Summary of input and output of Transformer models

Let's recap what we have learned so far about the input and output of Transformer models:

- **Input**: A sequence of tokens representing the input text.
- **Output**: A sequence of logits representing the next token predictions across all vocabulary at each position.

In the next two sections, we'll see how to use these logits to predict the next token.
"""
    )

    # Summary and Progress
    with st.container(border=True):
        st.markdown("**📚 Section Summary**")
        st.markdown(
            "Transformer models output logits - raw prediction scores for each vocabulary token at each position. These logits represent the model's confidence for potential next tokens."
        )

        # Progress indicator
        st.progress(2 / 5, text="Step 2/5: Logits Generation ✅")

    st.divider()

    return logits


def create_logits_heatmap(outputs, tokenizer, tokens):
    # Create facet grid: x-axis = sequence position, y-axis = vocabulary
    # Dynamically adjust top_k based on sequence length (minimum 10, maximum 20)
    top_k = max(5, 25 - (len(outputs) // 2))
    outputs_df = []
    for pos_idx, logits in enumerate(outputs):
        top_logits, top_indices = torch.topk(logits, k=top_k)
        for vocab_idx, logit_val in zip(top_indices.tolist(), top_logits.tolist()):
            outputs_df.append(
                {"position": pos_idx, "vocabulary": vocab_idx, "logit": logit_val}
            )

    df = pd.DataFrame(outputs_df)

    # Add token text for each position and calculate normalization per position
    df_with_tokens = []

    for pos_idx, logits in enumerate(outputs):
        pos_data = df[df["position"] == pos_idx]
        token_text = decode_token(tokenizer, tokens[pos_idx])

        # Get min/max from only the top N logits for this position
        if len(pos_data) > 0:
            position_min = pos_data["logit"].min()
            position_max = pos_data["logit"].max()
            position_range = position_max - position_min

            # Sort by logit descending to calculate ranks
            pos_data_sorted = pos_data.sort_values(
                "logit", ascending=False
            ).reset_index(drop=True)

            for rank, (_, row) in enumerate(pos_data_sorted.iterrows(), 1):
                # Calculate normalized logit using top N range only
                normalized_logit = (
                    (row["logit"] - position_min) / position_range
                    if position_range > 0
                    else 0
                )

                df_with_tokens.append(
                    {
                        "position": pos_idx,
                        "vocabulary": row["vocabulary"],
                        "logit": row["logit"],
                        "normalized_logit": normalized_logit,
                        "token_text": token_text,
                        "vocab_token": decode_token(tokenizer, int(row["vocabulary"])),
                        "rank": rank,
                        "position_token": f"{pos_idx}:{token_text}",
                        "prediction": f"{token_text} → {decode_token(tokenizer, int(row['vocabulary']))}",
                    }
                )
        else:
            # Add a placeholder entry to ensure all positions are represented
            df_with_tokens.append(
                {
                    "position": pos_idx,
                    "vocabulary": 0,  # Dummy vocabulary index
                    "logit": 0,
                    "normalized_logit": 0,
                    "token_text": token_text,
                    "vocab_token": decode_token(tokenizer, 0),
                    "rank": 1,  # Placeholder rank
                    "position_token": f"{pos_idx}:{token_text}",
                    "prediction": f"{token_text} → {decode_token(tokenizer, 0)}",
                }
            )

    chart_df = pd.DataFrame(df_with_tokens)

    # Create weather heatmap style chart with per-position normalization
    chart = (
        alt.Chart(chart_df)
        .mark_rect()
        .encode(
            x=alt.X(
                "vocabulary:O",
                title="Vocabulary Index",
                axis=alt.Axis(orient="top", labels=False, ticks=False),
            ),
            y=alt.Y(
                "position_token:N",
                title="",
                sort=alt.SortField(field="position", order="ascending"),
                axis=alt.Axis(
                    labelOverlap=True,
                    labelSeparation=1,
                    labelAlign="right",
                    labelBaseline="middle",
                    labelPadding=15,
                    tickCount={"expr": "length(domain('y'))"},
                    labelExpr="split(datum.label, ':')[1]",  # Show only the token part, not position
                ),
            ),
            color=alt.Color(
                "normalized_logit:Q",
                scale=alt.Scale(scheme="blues", domain=[0, 1]),
                legend=None,
            ),
            stroke=alt.value("white"),
            strokeWidth=alt.value(0.5),
            tooltip=[
                alt.Tooltip("position:O", title="Position"),
                alt.Tooltip("logit:Q", title="Logit"),
                alt.Tooltip("prediction:N", title="Prediction"),
                alt.Tooltip("vocabulary:O", title="Next Token Index"),
                alt.Tooltip("rank:O", title="Rank"),
            ],
        )
        .properties(
            height=len(outputs) * max(20, 70 - (len(outputs) - 1) * 5),
            title="Transformer Logits Heatmap",
        )
        .configure_axis(labelFontSize=10, titleFontSize=12)
        .configure_title(fontSize=14, offset=10)
    )

    st.altair_chart(chart, use_container_width=True)
