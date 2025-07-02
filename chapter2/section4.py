import streamlit as st
import torch
import inspect


def section4(weights):
    linear_weights = weights["lm_head.weight"]

    def linear(x):
        return torch.nn.functional.linear(x, linear_weights)

    return linear


def run(weights, tokenize, embedding):
    st.markdown(
        """
### Section 4: What is a Linear layer?
We've learned how the Transformer processes input tokens at the first layer. The next layer we're going to explore is the Linear layer, which is the last layer of the Transformer model. The Linear layer is responsible for converting the final embedding vectors into logits for each token.
"""
    )
    st.code(inspect.getsource(section4), line_numbers=True)
    linear = section4(weights)

    st.markdown(
        """
#### Linear transformation
At the end of the model, it needs to convert the final embedding vectors (1K-dimension) into logits (150K-dimension) because that's the final output of the model so that we can sample the next token based on the logit values.

Each dimension of the final embedding vector should have a different contribution to each token's logit. So, the basic idea to convert the embedding vector into logits is to sum all the dimensions of the embedding vector multiplied by some weights as shown below:
"""
    )
    st.code(
        f"""
# Embedding vector (1024-dimension)
x = [x₀, x₁, ..., x₁₀₂₃]

# Linear weights (151936 x 1024-dimension)
linear_weights = [
    [w₀₋₀     , w₀₋₁     , ..., w₀₋₁₀₂₃     ],
    [w₁₋₀     , w₁₋₁     , ..., w₁₋₁₀₂₃     ],
    ...,
    [w₁₅₁₉₃₅₋₀, w₁₅₁₉₃₅₋₁, ..., w₁₅₁₉₃₅₋₁₀₂₃]
]

# Logits (151936-dimension)
logits = [
    x₀*w₀₋₀ + x₁*w₀₋₁ + ... + x₁₀₂₃*w₀₋₁₀₂₃,
    x₀*w₁₋₀ + x₁*w₁₋₁ + ... + x₁₀₂₃*w₁₋₁₀₂₃,
    ...,
    x₀*w₁₅₁₉₃₅₋₀ + x₁*w₁₅₁₉₃₅₋₁ + ... + x₁₀₂₃*w₁₅₁₉₃₅₋₁₀₂₃
]"""
    )
    st.markdown(
        """
This is called a **Linear transformation** because it can be represented as a simple matrix multiplication. We won't describe matrix multiplication in detail here, but this can be simply done by the `torch.nn.functional.linear()` function as you can see in the `section4()` function above.
"""
    )

    st.markdown(
        """
#### Weight tying
In the previous section, we saw that the embedding weights are shared between the `embed_tokens` and `lm_head` (Linear) layers. This is called **weight tying**, i.e., the weights used by the linear layer are the same weights as the embedding layer.

With this design, the linear layer's output logits are basically the semantic similarity scores between the final embedding vector and each token. This is because the matrix multiplication calculates the dot product between them. As we saw above, the dot product is a way to measure similarity, and the more similar the two vectors are, the higher the dot product value is.

For example, assuming the final embedding vector is identical to the embedding vector of the token `world`, the logits vector will have higher values at the index of `world` and other similar tokens like `worlds`, `␣world`, etc.
"""
    )

    def logit(x, token):
        token_id = tokenize(token)[0]
        return linear(x)[token_id]

    x = embedding(tokenize("world"))[0]

    st.code(inspect.getsource(logit))
    st.code(
        f"""
# Assuming the model predicts "world"
x = embedding(tokenize("world"))[0]
logit(x, "world")  #=> {logit(x, "world"):.2f} (identical)
logit(x, "worlds") #=> {logit(x, "worlds"):.2f}
logit(x, " world") #=> {logit(x, " world"):.2f}
logit(x, "WORLD")  #=> {logit(x, " WORLD"):.2f}
logit(x, "math")   #=> {logit(x, " math"):.2f}"""
    )

    st.markdown(
        """
Therefore, if we sample the next token by following the logit values, the sampling distribution should be biased towards the embedding vector that is the model's prediction.
"""
    )

    # Summary and Progress
    with st.container(border=True):
        st.markdown("**📚 Section Summary**")
        st.markdown(
            "Linear layers convert final embedding vectors back to logits through matrix multiplication. Weight tying between embedding and linear layers creates semantic similarity scores for next token prediction."
        )
        
        # Progress indicator
        st.progress(4/5, text="Step 4/5: What is a Linear layer? ✅")

    st.divider()

    return linear
