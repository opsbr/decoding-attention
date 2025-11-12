import inspect
import streamlit as st
import torch
import altair as alt
import pandas as pd  # type: ignore
import numpy as np
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    MinPLogitsWarper,
)
from section1 import decode_token


def section3(logits, temperature, top_k, min_p, top_p):
    # Only consider the last token's logits for next token prediction
    last_logits = logits[-1]

    # Instantiate Hugging Face's logits warpers
    warpers = [
        TemperatureLogitsWarper(temperature=temperature),
        TopKLogitsWarper(top_k=top_k),
        MinPLogitsWarper(min_p=min_p),
        TopPLogitsWarper(top_p=top_p),
    ]

    # The warpers expect a batch of scores, so we use a batch of size = 1.
    # unsqueeze(0) is the same as [last_logits] but happens within PyTorch's tensor.
    scores = last_logits.unsqueeze(0)

    # Apply each warper
    for warper in warpers:
        scores = warper(torch.LongTensor(), scores)

    # Finally normalize scores to get probabilities of the next token
    # We only need the first batch element since we used a batch size of 1.
    # dim=-1 means we normalize across the last dimension (no meaning here but it's important for multi-dimensional tensor).
    return torch.nn.functional.softmax(scores[0], dim=-1)


def run(logits, tokenizer):
    st.markdown(
        """
### Section 3: Probability distribution - Temperature, Top-K, Min-P, Top-P, then Softmax

In this section, we will explain how to convert the logits we obtained in the previous section into a **probability distribution** over the next token. We'll first look into several techniques to adjust the logits, then describe why we use probabilities instead of logits. Here is the code we run to get the probabilities:
    """
    )
    st.code(inspect.getsource(section3), line_numbers=True)

    st.markdown(
        """
#### How to predict the next token?

First of all, we don't need the whole positions of logits. Since we just need the next token after the last position, **we only care about the last position's logits only**. The whole positions of logits are useful for training, but for inference we only need the last position.

##### Greedy decoding

Once we get the logits at the last position, the most naive way to predict the next token is to just pick the token with the highest logit value there. This is often called "Greedy decoding" and the code below shows how to do this with PyTorch's tensor:
"""
    )

    argmax_token = logits[-1].argmax().item()
    st.code(
        f"""
last_logits.argmax().item() # => {argmax_token} ({decode_token(tokenizer, argmax_token)})"""
    )

    st.markdown(
        """
This is the simplest way but it has some problems. For example, the prediction may get stuck in a loop e.g. _"the place that the place that ..."_. It also doesn't consider the remaining logit values at all, so it's too biased toward the highest logit value and loses the diversity of the next token possibilities. In general, humans don't use this method to speak or write text.
"""
    )

    st.markdown(
        """
##### Sampling decoding

Thus, in most cases, we use a stochastic method to pick the next token. This is often called "Sampling decoding" (or simply "Sampling"). Instead of picking the highest logit value, sampling picks the next token randomly based on the probability distribution. It still has the highest chance of picking the token with the maximum logit value, but it also allows picking other tokens, i.e., this is stochastic.

We'll talk about sampling in the next section, but before sampling, we want to adjust the logits to make them more suitable for sampling.
"""
    )

    st.markdown(
        """
#### Logits warping

Let's look into several techniques to adjust the logits before sampling. These techniques are often called "Logits warping" or "Logits processing". They are used to make the logits more suitable for sampling.

We use [Hugging Face's built-in logits warpers](https://huggingface.co/docs/transformers/v4.52.3/en/internal/generation_utils#transformers.AlternatingCodebooksLogitsProcessor) to apply these techniques. The Qwen3 model has its recommended sampling parameters [here](https://huggingface.co/Qwen/Qwen3-0.6B#best-practices): Temperature: `0.7`, Top-K: `20`, Min-P: `0.0`, Top-P: `0.8`.
"""
    )

    # Tabbed interface for warper explanations
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🌡️ Temperature", "🔢 Top-K", "📊 Min-P", "🎯 Top-P"]
    )

    with tab1:
        with st.container(border=True):
            st.markdown(
                """
            **Temperature Scaling** adjusts the sharpness of the distribution by a "temperature" parameter:
            - `temperature < 1.0`: More chances for higher logit tokens (sharper distribution)
                - Note: `~0.0` activates the highest logit token only (greedy)
            - `temperature = 1.0`: No scaling (original distribution)
            - `temperature > 1.0`: More chances for lower logit tokens (flatter distribution)
            """
            )

    with tab2:
        with st.container(border=True):
            st.markdown(
                """
            **Top-K Filtering** limits the number of tokens to consider:
            - Sort the logits in descending order and keep only the top K tokens
            - All other tokens get zero probability (filtered out)
            - Prevents sampling from very low-probability tokens
            """
            )

    with tab3:
        with st.container(border=True):
            st.markdown(
                """
            **Min-P Filtering** filters out tokens below a relative probability threshold:
            - `min_p = 0.0`: No filtering (all tokens considered)
            - `min_p > 0.0`: Filters out tokens with probabilities lower than `min_p × max(probabilities)`
            - Dynamically adjusts based on the highest probability token
            """
            )

    with tab4:
        with st.container(border=True):
            st.markdown(
                """
            **Top-P Filtering (Nucleus Sampling)** limits tokens to a cumulative probability mass:
            - `top_p = 1.0`: No filtering (all tokens considered)
            - `top_p < 1.0`: Keeps only tokens that make up the top `top_p` cumulative probability
            - Note: Internally calculates probabilities from logits first, then filters
            """
            )

    st.markdown(
        """
#### Softmax

The last step before sampling is to convert the logits into probabilities. Up until here, we have been working with logits, which are raw scores that can be positive or negative. However, to sample the next token stochastically, we need **probability distribution** instead. Probability distribution has the following two properties:

1. All values are non-negative.
2. The sum of all values is `1.0`.

To convert logits to probability distribution, there is a very commonly used function called [**Softmax**](https://en.wikipedia.org/wiki/Softmax_function). It takes arbitrary logits and normalizes them to a probability distribution that satisfies the above two properties. It has a mathematical formula but we don't care the details. We just use PyTorch's built-in function [`torch.nn.functions.softmax()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html) to do this for us.

The below is the final output of this section:
"""
    )

    temperature, top_k, min_p, top_p = sidebar_sampling_parameters()
    probabilities = section3(logits, temperature, top_k, min_p, top_p)
    st.code(
        f"""
# Almost all tokens are 0.0 because of Top-K
probabilities = {probabilities}
probabilities.shape = {probabilities.shape}"""
    )

    st.markdown(
        """
#### Probability distribution visualization

Finally, we visualize the probability distribution of the next token. The x-axis is the probability (`0.0` to `1.0`), and the y-axis is the token. "Filtered" means the probabilities after applying the logits warping techniques, and "Original" means the probabilities before filtering.

⬅️Change sampling parameters in the sidebar. The default values are the same as Qwen3's recommended values.
"""
    )
    chart = create_probability_visualization(logits, probabilities, tokenizer, viz_k=10)
    st.altair_chart(chart, width="stretch")

    # Summary and Progress
    with st.container(border=True):
        st.markdown("**📚 Section Summary**")
        st.markdown(
            "Logits are transformed into probability distributions using sampling filters (temperature, top-k, min-p, top-p) and softmax normalization. This creates a proper probability distribution for stochastic sampling."
        )

        # Progress indicator
        st.progress(3 / 5, text="Step 3/5: Probability Distribution ✅")

    st.divider()

    return probabilities


def sidebar_sampling_parameters():
    # Initialize session state for sampling parameters
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "top_k" not in st.session_state:
        st.session_state.top_k = 20
    if "min_p" not in st.session_state:
        st.session_state.min_p = 0.0
    if "top_p" not in st.session_state:
        st.session_state.top_p = 0.8

    # Compact sampling parameters in sidebar accordion
    with st.sidebar.expander("🎛️ Sampling Parameters", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.session_state.temperature = st.slider(
                "Temperature", 0.1, 2.0, st.session_state.temperature, 0.1
            )
            st.session_state.min_p = st.slider(
                "Min-P", 0.0, 0.5, st.session_state.min_p, 0.01
            )

        with col2:
            st.session_state.top_k = st.slider(
                "Top-K", 1, 100, st.session_state.top_k, 5
            )
            st.session_state.top_p = st.slider(
                "Top-P", 0.1, 1.0, st.session_state.top_p, 0.05
            )

    temperature = st.session_state.temperature
    top_k = st.session_state.top_k
    min_p = st.session_state.min_p
    top_p = st.session_state.top_p

    return temperature, top_k, min_p, top_p


def create_probability_visualization(
    logits, probabilities, tokenizer, viz_k=10, sampled_token=None
):
    # Get top tokens based on original probabilities (before filtering)
    original_probs = torch.nn.functional.softmax(logits[-1], dim=-1)
    top_original_probs, display_indices = torch.topk(original_probs, k=viz_k)

    # Get the filtered probabilities for these same tokens
    filtered_probs = probabilities[display_indices]

    # Convert to numpy and ensure no NaN/inf values
    top_original_probs_np = top_original_probs.detach().numpy()
    filtered_probs_np = filtered_probs.detach().numpy()

    # Replace any NaN or inf values with 0
    top_original_probs_np = np.nan_to_num(
        top_original_probs_np, nan=0.0, posinf=0.0, neginf=0.0
    )
    filtered_probs_np = np.nan_to_num(
        filtered_probs_np, nan=0.0, posinf=0.0, neginf=0.0
    )

    # Create token representations with fallback for Unicode display
    token_labels = []
    for idx in display_indices:
        try:
            token = decode_token(tokenizer, int(idx.item()), max_length=15)
            # Replace problematic characters that might cause rendering issues
            if not token.strip():
                token = f"[{int(idx.item())}]"
            token_labels.append(token)
        except:
            token_labels.append(f"[{int(idx.item())}]")

    # Create comparison DataFrame and sort by original probability descending (highest first)
    df = pd.DataFrame(
        {
            "Token": token_labels,
            "Original_Probability": top_original_probs_np,
            "Filtered_Probability": filtered_probs_np,
            "Token_ID": [int(idx.item()) for idx in display_indices],
        }
    ).sort_values("Original_Probability", ascending=False)

    # Reset index to get proper y-position (0 = highest probability)
    df = df.reset_index(drop=True)

    # Create data for side-by-side grouped bar chart
    chart_data = []
    for i, row in df.iterrows():
        orig_prob = float(row["Original_Probability"])
        filtered_prob = float(row["Filtered_Probability"])
        token = str(row["Token"])
        token_id = int(row["Token_ID"])

        # Check if this token was sampled
        is_sampled = sampled_token is not None and token_id == sampled_token

        # Ensure finite values
        if not np.isfinite(orig_prob):
            orig_prob = 0.0
        if not np.isfinite(filtered_prob):
            filtered_prob = 0.0

        # Add both original and filtered as separate rows for side-by-side display
        chart_data.append(
            {
                "token": token,
                "probability": orig_prob,
                "type": "Original",
                "order": i,
                "sampled": is_sampled,
            }
        )

        chart_data.append(
            {
                "token": token,
                "probability": filtered_prob,
                "type": "Filtered",
                "order": i,
                "sampled": is_sampled,
            }
        )

    chart_df = pd.DataFrame(chart_data)

    # Get the sampled token name for conditional formatting
    sampled_token_name = None
    if sampled_token is not None:
        sampled_rows = chart_df[chart_df["sampled"] == True]
        if not sampled_rows.empty:
            sampled_token_name = sampled_rows.iloc[0]["token"]

    # Create the main bar chart with conditional y-axis label coloring
    bars = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "probability:Q",
                title="Probability",
                scale=alt.Scale(nice=False, padding=0.1),
            ),
            y=alt.Y(
                "token:N",
                sort=alt.EncodingSortField(field="order", order="ascending"),
                title="Next Token",
                axis=alt.Axis(
                    labelColor=alt.condition(
                        (
                            f"datum.label == '{sampled_token_name}'"
                            if sampled_token_name
                            else "false"
                        ),
                        alt.value("red"),
                        alt.value("black"),
                    )
                ),
            ),
            color=alt.Color(
                "type:N",
                scale=alt.Scale(
                    domain=["Filtered", "Original"], range=["darkred", "lightgray"]
                ),
                title=None,
                legend=alt.Legend(orient="bottom-right"),
            ),
            yOffset=alt.YOffset("type:N"),
        )
    )

    # Add text label for filtered probability value on sampled token
    text_labels = (
        alt.Chart(chart_df)
        .mark_text(align="left", dx=5, fontSize=12, fontWeight="bold", color="darkred")
        .encode(
            x=alt.X("probability:Q"),
            y=alt.Y(
                "token:N", sort=alt.EncodingSortField(field="order", order="ascending")
            ),
            text=alt.condition(
                "datum.sampled && datum.type == 'Filtered'",
                alt.Text("probability:Q", format=".2f"),
                alt.value(""),
            ),
            yOffset=alt.YOffset("type:N"),
        )
    )

    # Combine bars and text labels
    chart = (
        (bars + text_labels)
        .properties(width=600, height=300, title="Next Token Probability Distribution")
        .resolve_scale(x="shared")
    )

    return chart
