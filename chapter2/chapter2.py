import cache
import streamlit as st
import inspect


from chapter2 import section1, section2, section3, section4, section5


def chapter2():
    # Model configuration
    model_name = "Qwen/Qwen3-0.6B"

    # Section 1: Recap of Chapter 1 - tokenization and sampling
    pretrained, tokenize, decode, sample = section1.run(model_name)

    # Section 2: Explore pretrained weights and model architecture
    weights = section2.run(pretrained)

    # Section 3: Embedding layer - converting token IDs to dense vectors
    embedding = section3.run(weights, tokenize)

    # Section 4: Linear layer - converting embeddings back to logits
    linear = section4.run(weights, tokenize, embedding)

    # Section 5: Combine embedding and linear layers for next token prediction
    section5.run(embedding, linear, tokenize, decode, sample)


st.markdown(
    """
## Chapter 2: Embedding & Linear
In this chapter, we'll dive deeper into the Transformer architecture by exploring the first and last layers of the model.
    """
)
# Add high-level diagram at the beginning
st.markdown("### 🔄 Learning Overview")
st.markdown(
    "This chapter covers the embedding and linear layers that are the first and last components of the Transformer architecture:"
)

# Display overview image
st.image("chapter2/overview.png", use_container_width=True)

st.markdown(
    """
Here is the overall flow of this chapter:
    """
)
st.code(inspect.getsource(chapter2), line_numbers=True)

# Add section navigation links to sidebar
st.sidebar.markdown(
    """
### 📖 Sections
1. [Recap Chapter 1](#section-1-recap-chapter-1)
2. [Pretrained Weights](#section-2-pretrained-weights)
3. [What is Embedding?](#section-3-what-is-embedding)
4. [Linear Layer](#section-4-what-is-linear-layer)
5. [Stitch Together](#section-5-stitch-together)
"""
)
st.sidebar.divider()

st.divider()
chapter2()

st.markdown(
    """
## Conclusion of Chapter 2

This concludes Chapter 2. You have learned about the first and last layers of the Transformer architecture - the embedding and linear layers. You now understand how token IDs are converted into meaningful vector representations and how those vectors are transformed back into logits for next token prediction.

However, our current model is quite limited. It simply passes embedding vectors directly to the linear layer, resulting in predictions that only consider the last input token while ignoring all previous context. This is where the core Transformer architecture becomes essential.
"""
)

# Add diagram at conclusion showing what was learned
st.markdown("### 🎯 What You've Mastered")
st.markdown(
    "You now understand the input and output layers of the Transformer, but the middle is still TODO:"
)

# Display overview image again at conclusion
st.image("chapter2/overview.png", use_container_width=True)

st.markdown(
    """
In the next chapter, we will explore the core Transformer processing that happens between the embedding and linear layers - first the neural network layer, then the attention mechanism.
"""
)
