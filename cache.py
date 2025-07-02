import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM


@st.cache_resource
def tokenizer_from_pretrained(model_name):
    return original_tokenizer_from_pretrained(model_name)


original_tokenizer_from_pretrained = AutoTokenizer.from_pretrained
AutoTokenizer.from_pretrained = tokenizer_from_pretrained  # type: ignore


@st.cache_resource
def model_from_pretrained(model_name):
    return original_model_from_pretrained(model_name)


original_model_from_pretrained = AutoModelForCausalLM.from_pretrained
AutoModelForCausalLM.from_pretrained = model_from_pretrained  # type: ignore
