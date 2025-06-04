import inspect
import streamlit as st
import torch
from section1 import create_tokenization_visualization


def section4(probabilities, current_tokens):
    # Sample one token from the given probabilities
    sampled_token = torch.multinomial(probabilities, num_samples=1).item()
    # Append the sampled token to the current tokens
    new_tokens = current_tokens + [sampled_token]
    return sampled_token, new_tokens


def run(probabilities, tokens, tokenizer):
    st.markdown(
        """
### Section 4: Sampling the next token

In this section, we will sample the next token from the probability distribution obtained from the logits. The steps are pretty straightforward like below:
    """
    )
    st.code(inspect.getsource(section4), line_numbers=True)

    # Initialize session state for sampling
    if "current_sampled_token" not in st.session_state:
        st.session_state.current_sampled_token = None
        st.session_state.current_new_tokens = None
        st.session_state.force_resample = True
        st.session_state.last_input_text = ""

    # Check if input text changed - if so, force resample
    current_input = st.session_state.get("input_text", "")
    if current_input != st.session_state.last_input_text:
        st.session_state.force_resample = True
        st.session_state.last_input_text = current_input

    # Force resample if first time
    if st.session_state.force_resample:
        sampled_token, new_tokens = section4(probabilities, tokens)
        st.session_state.current_sampled_token = sampled_token
        st.session_state.current_new_tokens = new_tokens
        st.session_state.force_resample = False
    else:
        # Use existing samples
        sampled_token = st.session_state.current_sampled_token
        new_tokens = st.session_state.current_new_tokens

    # Store sampled token in session state for use in visualizations
    st.session_state.sampled_token = sampled_token

    st.markdown(
        """
#### Multinomial Sampling

Since we already have the probability distribution for the next token, we can simply use [`torch.multinomial()`](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html) to sample one token from this distribution. This function takes the probabilities, then samples one token based on the distribution. We don't care about the mathematical details but it simply picks one token based on the probabilities, so the higher the probability, the more likely it is to be chosen.
"""
    )

    st.markdown(
        """
#### One sampling exercise

Here is the result of one sampling from the given probability distribution. Hope the new tokens looks like a continuation of your input text!
"""
    )

    st.markdown("**Sampled Token**")
    create_tokenization_visualization([sampled_token], tokenizer)

    st.markdown("**New Tokens**")
    create_tokenization_visualization(new_tokens, tokenizer)

    st.code(
        f"""
{sampled_token = }
{new_tokens = }"""
    )

    st.markdown(
        """
You can resample the next token from the same probability distribution by clicking the button below. This will generate a new random sample, which may or may not be the same as the previous one.
"""
    )

    # Resample button at bottom
    if st.button(
        "🎲 Resample Token",
        key="resample_btn",
        help="Generate a new random sample from the same probability distribution",
    ):
        # Force resample on next run
        st.session_state.force_resample = True

    # Summary and Progress
    with st.container(border=True):
        st.markdown("**📚 Section Summary**")
        st.markdown("Multinomial sampling selects the next token from the probability distribution. Higher probability tokens are more likely to be chosen, but randomness ensures diverse generation.")
        
        # Progress indicator
        st.progress(4/5, text="Step 4/5: Token Sampling ✅")

    st.divider()

    return sampled_token, new_tokens
