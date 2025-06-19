import streamlit as st
import torch
import os

# To suppress the error below.
# RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_
torch.classes.__path__ = []

# Hard-reset the only legal CPU dtype to avoid the error below.
# RuntimeError: unsupported scalarType
torch.set_autocast_dtype("cpu", torch.bfloat16)

# Configure page
st.set_page_config(
    page_title="Decoding Attention",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title(":blue[_**Decoding Attention**_]")
st.logo("logo.png", size="large")


def readme_page():
    """Load README.md content"""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
        st.markdown(content)


# Define pages with descriptive titles
pages = {
    "Decoding Attention": [
        st.Page(readme_page, title="README", icon="📖"),
        st.Page(
            "./chapter1/chapter1.py",
            title="Chapter 1: Tokenization & Sampling",
            icon="📚",
        ),
    ]
}

# Create navigation with grouped pages
nav = st.navigation(pages)

# Run the selected page
nav.run()
