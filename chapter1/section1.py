import inspect
import streamlit as st
import pandas as pd  # type: ignore
import cache
from transformers import AutoTokenizer
from annotated_text import annotated_text  # type: ignore


def section1(model_name, input_text):
    # Instantiate BPE tokenizer with pretrained data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer() returns a dict with "input_ids", "attention_mask", etc.
    # We only need the "input_ids"(= tokens) for this example.
    tokenized = tokenizer(input_text)
    return tokenizer, tokenized.input_ids


def run(model_name):
    st.markdown(
        """
### Section 1: Tokenizer - Byte Pair Encoding (BPE)

This section describes the input to the Transformer. Your input to an LLM is typically a (UTF-8) text like "Input text" on the left sidebar. However, **Transformers are neural networks that only process numerical data**, so we need to convert text into numbers. This is done by a **Tokenizer**, which converts text into tokens (numbers).

We're going to instantiate a tokenizer and process the input text to tokens. See the snippet below:
    """
    )
    sidebar_input_text()

    st.code(inspect.getsource(section1), line_numbers=True)
    tokenizer, tokens = section1(model_name, st.session_state.input_text)

    st.markdown(
        """
#### How tokens look like?

One naive method is to split text into words, but this is less resilient to typos or new vocabularies. Another method is to split text into characters, but this makes sequences too long. We'll talk about this later, but the length of the input sequence is a very expensive resource for Transformers. So, neither is a good solution so far.

```python
# Split into words
"Hello world" => ["Hello", "world"] => [500, 501] # L = 2👍
"Hello worl"  => ["Hello", "worl"]  => [500, ???]👎

# Split into chars (or bytes)
"Hello world" => [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100] # L = 11👎
```
"""
    )

    st.markdown(
        """
##### Byte Pair Encoding (BPE)
The most common tokenization method is called [**Byte Pair Encoding (BPE)**](https://en.wikipedia.org/wiki/Byte-pair_encoding), which splits text into subword units. BPE tokenizer requires training phase to learn the vocabulary (i.e. the list of all tokens) from a training dataset. Because Transformer is trained based on a specific vocabulary, we should use the same tokenizer that was used when training Qwen3 models. Qwen3's tokenizer data is provided by [Hugging Face](https://huggingface.co/docs/transformers/main_classes/tokenizer), so we just loaded the pretrained tokenizer. Here is the sample tokens of Qwen3 tokenizer:
"""
    )
    create_vocabulary_overview(tokenizer)

    st.markdown(
        """
As you can see above, each token represents several bytes sequence e.g. `ID: 500` is `pl` (or `70 6c`). How to train this mapping is not trivial, so we won't explain it here, but the basic idea is: 1. Split text into bytes, 2. Merge frequently occurring byte pairs into new tokens, and 3. Repeat until the vocabulary size reaches the desired size. So, the more frequent byte pairs appear in the training dataset, the more likely they will be merged into a single token, and this happens recursively. Because all 256 bytes are included, any text can be tokenized because (utf-8) text ends up to a sequence of bytes.
"""
    )

    st.markdown(
        """
#### Tokenize Input text

Now, let's tokenize the input text. The tokenizer will convert the input text into tokens (integers) so that Transformer can understand it. You can see the tokenization results below.

⬅️Feel free to edit **Input text** in the left sidebar text area. Then, the tokens below will be updated automatically.
                """
    )
    with st.container(border=True):
        create_tokenization_visualization(tokens, tokenizer)
    st.caption(
        "This visualization is inspired by [kspviswa/TikTokenViewer](https://github.com/kspviswa/TikTokenViewer) and built by [tvst/st-annotated-text](https://github.com/tvst/st-annotated-text)."
    )

    st.code(
        f"""
input_text  = "{st.session_state.input_text}"
tokens      = {tokens}
len(tokens) = {len(tokens)}
""",
        wrap_lines=True,
    )

    # Summary and Progress
    with st.container(border=True):
        st.markdown("**📚 Section Summary**")
        st.markdown(
            "Tokenization converts text into numerical tokens using Byte Pair Encoding (BPE). Each token represents a subword unit, enabling efficient text processing by Transformers."
        )

        # Progress indicator
        st.progress(1 / 5, text="Step 1/5: Tokenization ✅")

    st.divider()

    return tokenizer, tokens


def decode_token(tokenizer, token_id: int, max_length: int = 32):
    """
    Decode a token ID to a clean, display-ready string.
    Handles non-printable characters and applies consistent formatting.
    """
    try:
        # Use tokenizer.decode() for proper decoding
        decoded_token = tokenizer.decode([token_id])

        # Replace spaces with visible character
        if decoded_token == " ":
            display_token = "␣"
        elif decoded_token.startswith(" "):
            display_token = "␣" + decoded_token[1:]
        else:
            display_token = decoded_token

        # Handle non-printable characters and special Unicode characters
        clean_chars = []
        for char in display_token:
            char_code = ord(char)
            # Handle variation selectors and other invisible Unicode characters
            if char_code in [0xFE0F, 0xFE0E]:  # Variation selectors
                clean_chars.append(f"U+{char_code:04X}")
            elif 0x200B <= char_code <= 0x200F:  # Zero-width characters
                clean_chars.append(f"U+{char_code:04X}")
            elif 0x2060 <= char_code <= 0x206F:  # More invisible formatting characters
                clean_chars.append(f"U+{char_code:04X}")
            elif (
                char.isprintable() and char.strip()
            ):  # Printable and not just whitespace
                clean_chars.append(char)
            elif char == "\n":
                clean_chars.append("\\n")
            elif char == "\t":
                clean_chars.append("\\t")
            elif char == "\r":
                clean_chars.append("\\r")
            else:
                # Replace other non-printable with unicode escape
                clean_chars.append(f"U+{char_code:04X}")

        display_token = "".join(clean_chars)

        # Truncate if too long
        if len(display_token) > max_length:
            display_token = display_token[: max_length - 3] + "..."

        return display_token

    except:
        # Fallback for any decode errors
        return f"<err:{token_id}>"


def sidebar_input_text():
    # Initialize session state if needed
    if "input_text" not in st.session_state:
        st.session_state.input_text = "Hello world!"

    # Check if autoregressive is active
    autoregressive_active = st.session_state.get("autoregressive_started", False)

    # Direct text area input
    st.sidebar.text("Input text:")

    # Use the widget key directly for state management when not in autoregressive mode
    if not autoregressive_active:
        # Initialize widget state if needed
        if "input_text_area" not in st.session_state:
            st.session_state.input_text_area = st.session_state.input_text

        new_text = st.sidebar.text_area(
            label="Input text:",
            max_chars=200,
            height=100,
            key="input_text_area",
            label_visibility="collapsed",
        )

        # Update main session state from widget state
        st.session_state.input_text = new_text
    else:
        # When autoregressive is active, show read-only version
        st.sidebar.text_area(
            label="Input text:",
            value=st.session_state.input_text,
            max_chars=200,
            height=100,
            disabled=True,
            label_visibility="collapsed",
        )


def create_vocabulary_overview(tokenizer):
    vocab_dict = tokenizer.get_vocab()
    total_vocab = len(vocab_dict)

    # Sample tokens at regular intervals
    sample_interval = 500

    # Get sampled tokens
    vocab_items = sorted(vocab_dict.items(), key=lambda x: x[1])
    sampled_tokens = []

    for i in range(0, total_vocab, sample_interval):
        if i < len(vocab_items):
            token, token_id = vocab_items[i]
            display_token = decode_token(tokenizer, token_id)
            # Get bytes representation
            try:
                decoded_bytes = tokenizer.decode([token_id]).encode("utf-8")
                bytes_repr = " ".join(f"{b:02x}" for b in decoded_bytes)
            except:
                bytes_repr = "N/A"
            sampled_tokens.append((token_id, display_token, bytes_repr))

    # Create 3-column table
    table_data = []
    for token_id, token_text, bytes_repr in sampled_tokens:
        table_data.append({"ID": token_id, "Token": token_text, "Bytes": bytes_repr})

    df = pd.DataFrame(table_data)

    # Display as table with styling
    styled_df = (
        df.style.set_properties(
            **{
                "background-color": "#f0f2f6",
                "color": "black",
                "font-family": "monospace",
            },
            subset=["ID"],
        )
        .set_properties(
            **{
                "background-color": "#e8f4fd",
                "color": "black",
                "font-family": "monospace",
            },
            subset=["Token"],
        )
        .set_properties(
            **{
                "background-color": "#fff2cc",
                "color": "black",
                "font-family": "monospace",
            },
            subset=["Bytes"],
        )
    )

    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        height=200,
        column_config={
            "ID": st.column_config.TextColumn(label="ID", width="small"),
            "Token": st.column_config.TextColumn(label="Token", width="medium"),
            "Bytes": st.column_config.TextColumn(label="Bytes", width="large"),
        },
    )
    st.caption(
        f"Showing every {sample_interval}th token • Total vocabulary: {total_vocab:,}"
    )


def create_tokenization_visualization(
    tokens, tokenizer, max_length=None, align_with_length=None, align_end_offset=0
):
    # Use align_with_length if provided to determine omission point consistently
    effective_max = align_with_length if align_with_length is not None else max_length

    if max_length is None or len(tokens) <= max_length:
        # Show all tokens if no limit or within limit
        annotated_text([(decode_token(tokenizer, t), str(t)) for t in tokens])
    elif effective_max is not None and len(tokens) > effective_max:
        # Show only last tokens with ellipsis at the beginning
        # Calculate how many tokens to show from the end, accounting for alignment offset
        end_count = effective_max + align_end_offset
        end_tokens = (
            tokens[-end_count:]
            if end_count > 0 and end_count <= len(tokens)
            else tokens
        )

        # Create visualization with ellipsis at the beginning
        ellipsis_text = " ...... "  # Plain text, not a tuple
        end_annotations = [
            (decode_token(tokenizer, t, max_length=8), str(t)) for t in end_tokens
        ]

        annotated_text([ellipsis_text] + end_annotations)
    else:
        # Fallback to show only last tokens
        end_tokens = tokens[-max_length:]

        ellipsis_text = " ...... "
        end_annotations = [
            (decode_token(tokenizer, t, max_length=8), str(t)) for t in end_tokens
        ]

        annotated_text([ellipsis_text] + end_annotations)
