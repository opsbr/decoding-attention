import streamlit as st
import torch
import inspect


def section5(input, embedding, linear, tokenize, decode, sample):
    tokens = tokenize(input)

    x = embedding(tokens)
    # TODO: We'll work here in the following chapters.
    logits = linear(x)

    next_token = sample(logits)
    return decode(next_token)


def run(embedding, linear, tokenize, decode, sample):
    st.markdown(
        """
### Section 5: Stitch together
Now, we've learned about the first and last layers of the model. Let's stitch them together to see how our current next token prediction works.
"""
    )
    st.code(inspect.getsource(section5), line_numbers=True)

    st.markdown(
        """
#### Current next token prediction
Since we haven't implemented any core logic of the Transformer yet, our prediction doesn't work well. But let's see how our current model predicts the next token. Here are the results of 10 experiments (the randomness is from the `sample()` function):
"""
    )

    input = "Hello world"
    results = [
        section5(input, embedding, linear, tokenize, decode, sample) for _ in range(10)
    ]
    st.code(
        f"""
input = "{input}" # ({tokenize(input)})
results = [
    section5(input, embedding, linear, tokenize, decode, sample)
    for _ in range(10)
]"""
    )
    st.write(results)

    st.markdown(
        """
As you can see, the model always predicts something similar to ` world`. Why? This is because we directly pass the input embedding vectors to the linear layer, so the model is predicting the input tokens just like we saw in the previous section.

Since the last position's input token is `world`, the model predicts `world`-like tokens as the next token and doesn't care about the other input tokens at all yet.

"""
    )

    # Summary and Progress
    with st.container(border=True):
        st.markdown("**📚 Section Summary**")
        st.markdown(
            "We combined embedding and linear layers into a basic model pipeline. Without Transformer processing, the model simply predicts tokens similar to the last input token, ignoring context from earlier positions."
        )
        
        # Progress indicator
        st.progress(5/5, text="Step 5/5: Stitch Together ✅")

    st.divider()
