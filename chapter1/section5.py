import inspect
import streamlit as st
from section1 import create_tokenization_visualization
from section3 import create_probability_visualization


def run(tokens, logits, probabilities, sampled_token, new_tokens, tokenizer):
    st.markdown(
        """
### Section 5: Autoregressive Inference

The final section of this chapter is about autoregressive inference, where we generate tokens one by one based on the previous tokens. This is how most language models work in practice, generating text in a sequential manner. You typically see the streaming of text generation when you use LLM applications and this is the core of that process.

#### Autoregressive Generation

Autoregressive generation means that we repeat the process by feeding the new tokens i.e. the previous tokens + the next token generated back into the model again. In our case, we already generated the new tokens in the previous section. So, we only need to update the input text with the new tokens and run the process again.

⬇️Toggle the "Autoregressive Generation" switch below to start autoregressive generation.
    """
    )
    # Initialize autoregressive state
    if "autoregressive_started" not in st.session_state:
        st.session_state.autoregressive_started = False

    # Autoregressive toggle
    autoregressive_active = st.toggle(
        "🚀 Autoregressive Generation",
        value=st.session_state.autoregressive_started,
        key="autoregressive_toggle",
        help="Enable continuous token generation",
    )

    # Handle state changes
    if autoregressive_active != st.session_state.autoregressive_started:
        st.session_state.autoregressive_started = autoregressive_active
        if autoregressive_active and not st.session_state.get("show_edit_modal", False):
            # Update text when first toggled on
            new_text = tokenizer.decode(new_tokens)
            st.session_state.input_text = new_text

    # Show visualization if autoregressive is active
    show_visualization = st.session_state.autoregressive_started

    # Only show tokens and probabilities after autoregressive generation has started
    if show_visualization:
        st.markdown(
            """
#### Visualizing Autoregressive Generation

You'll see the original tokens at the top followed by the next token probabilities with highlighted the sampled token, and the next tokens at the bottom. During the autoregressive generation, you might see a special token like `<|endoftext|>` at the end of the sequence, which indicates the end of the text generation and people usually stop generation there, or at the maximum length of the sequence.

⬇️Click "Generate Next Token" to experience the autoregressive generation process step by step.
"""
        )

        # Original tokens
        with st.container(border=True, height=150):
            create_tokenization_visualization(
                tokens, tokenizer, max_length=20, align_with_length=20
            )

        # Display probability visualization with sampled token highlighted
        viz_k = min(st.session_state.get("top_k", 20), 10)
        chart = create_probability_visualization(
            logits, probabilities, tokenizer, viz_k, sampled_token
        )
        st.altair_chart(chart, use_container_width=True)

        # New tokens after sampling
        with st.container(border=True, height=150):
            create_tokenization_visualization(
                new_tokens,
                tokenizer,
                max_length=21,
                align_with_length=20,
                align_end_offset=1,
            )

    # Manual generation button when autoregressive is active
    if st.session_state.autoregressive_started:
        generate_clicked = st.button(
            "🔄 Generate Next Token",
            key="manual_generate_btn",
            help="Generate the next token manually",
        )

        if generate_clicked and not st.session_state.get("show_edit_modal", False):
            new_text = tokenizer.decode(new_tokens)
            st.session_state.input_text = new_text

        st.caption(
            "In practice, the next tokens are directly fed back into the model instead of decoding them to text."
        )

    # Summary and Progress
    with st.container(border=True):
        st.markdown("**📚 Section Summary**")
        st.markdown("Autoregressive generation repeats the entire process by feeding generated tokens back into the model. This creates sequential text generation where each new token depends on all previous tokens.")
        
        # Progress indicator
        st.progress(5/5, text="Step 5/5: Autoregressive Generation ✅")

    st.divider()
