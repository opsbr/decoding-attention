import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import inspect

torch.classes.__path__ = []


def demo():
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Vocaburary size is the number of unique tokens in the tokenizer.
    # The second term is for the special tokens like <|endoftext|>.
    vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens_decoder)

    st.text(f"{vocab_size = :,}")

    input_text = st.text_input("input_text", "Hello, world!")

    st.write(input_text)


st.code(inspect.getsource(demo), language="python")

demo()
