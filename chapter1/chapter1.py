import inspect
import streamlit as st
import sys
import os

# Add current directory to Python path for section imports
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import sections
import section1
import section2
import section3
import section4
import section5


def chapter1():
    # Model configuration
    model_name = "Qwen/Qwen3-0.6B"

    # Section 1: Tokenizer and BPE
    tokenizer, tokens = section1.run(model_name)

    # Section 2: Logits inference with Transformer
    logits = section2.run(model_name, tokens, tokenizer)

    # Section 3: Probability distribution of the next token
    probabilities = section3.run(logits, tokenizer)

    # Section 4: Sampling the next token
    sampled_token, new_tokens = section4.run(probabilities, tokens, tokenizer)

    # Section 5: Autoregressive inference
    section5.run(tokens, logits, probabilities, sampled_token, new_tokens, tokenizer)


st.markdown(
    """
## Chapter 1: Tokenization and Sampling
In this chapter, we'll talk about what's outside of the Transformer itself, i.e., the input and output of the Transformer. Once you understand what the input and output of the Transformer are, you can easily imagine how the Transformer works even if you don't know the details of the Transformer architecture yet. I think this is the most natural way to learn new software as a software engineer.
    """
)

# Add high-level diagram at the beginning
st.markdown("### 🔄 Learning Overview")
st.markdown(
    "This chapter covers the complete pipeline from text input to text output, focusing on everything **outside** the Transformer:"
)

# Display overview image
st.image("chapter1/overview.png", use_container_width=True)

st.markdown(
    """
Here is the overall Flow of this chapter:
    """
)
st.code(inspect.getsource(chapter1), line_numbers=True)

# Add section navigation links to sidebar
st.sidebar.markdown(
    """
### 📖 Sections
1. [Tokenizer](#section-1-tokenizer-byte-pair-encoding-bpe)
2. [Transformer Inference](#section-2-logits-inference-with-transformer)
3. [Probability Distribution](#section-3-probability-distribution-temperature-top-k-min-p-top-p-then-softmax)
4. [Sampling](#section-4-sampling-the-next-token)
5. [Autoregression](#section-5-autoregressive-inference)
"""
)
st.sidebar.divider()

st.divider()
chapter1()

st.markdown(
    """
## Conclusion of Chapter 1

This concludes Chapter 1. You have learned about the input and output of Transformer models, how to tokenize text, and how to sample the next token based on the model's predictions. You also experienced the autoregressive generation process, which is how most language models generate text in practice.
"""
)

# Add diagram at conclusion showing what was learned
st.markdown("### 🎯 What You've Mastered")
st.markdown(
    "You now understand the complete pipeline around the Transformer - everything except the **🤖 Transformer** itself:"
)

# Display overview image again at conclusion
st.image("chapter1/overview.png", use_container_width=True)

st.markdown(
    """
In the next chapter, we will dive one level deeper into the Transformer architecture and how it processes these inputs to generate outputs.

##### Postscripts
"""
)

with st.expander("**Why do I start with Tokenization and Sampling?**"):
    st.markdown(
        """
Majority of Transformer explanations skip these area because **Tokenization and Sampling are not Transformer-specific topics** and they are well-known in machine learning (ML) or natural language processing (NLP) communities. Most of the contents focus on Attention and other Transformer-specific topics as these are their interests.

However, I was confused while learning Transformers because I couldn't imagine how the input and output look like until I exercised **Stanford CS336's assignment 1**. This is the initial motivation of this chapter. Software engineers love to treat unknown software as long as the input and output are manageable such as HTTP APIs. Now, you know that Transformer ends up to just a numerical function.
"""
    )
