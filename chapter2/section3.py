import streamlit as st
import torch
import inspect
import pandas as pd  # type: ignore
import altair as alt


def section3(weights):
    embedding_weights = weights["model.embed_tokens.weight"]

    def embedding(input):
        return torch.nn.functional.embedding(input, embedding_weights)

    return embedding


def run(weights, tokenize):
    st.markdown(
        """
### Section 3: What is embedding?
Now, let's use the first weight to calculate the first layer of the model, which is the embedding layer. The embedding layer is responsible for converting token IDs into dense vector representations (embeddings) that capture semantic meaning:
"""
    )
    st.code(inspect.getsource(section3), line_numbers=True)
    embedding = section3(weights)

    st.markdown(
        """
#### Representing tokens as embedding vectors
The input of the model is a sequence of token IDs e.g. `[1,2,3]`, but this token ID representation can't capture the semantic meaning of the tokens. For example, the model doesn't know that `world` and `␣world` tokens are similar to each other based solely on token IDs:
"""
    )
    st.code(
        f"""
tokenize("world")  #=> {tokenize("world")}
tokenize(" world") #=> {tokenize(" world")}"""
    )

    st.markdown(
        """
An embedding vector is a way to represent tokens in a high-dimensional space where similar tokens are closer together.
"""
    )
    st.code(
        f"""
embedding(tokenize("world")) #=> {embedding(tokenize("world"))}
embedding(tokenize(" world")) #=> {embedding(tokenize(" world"))}"""
    )
    with st.expander("_What is vector?_"):
        st.markdown(
            """
            A vector is just a mathematical term that refers to a 1D array of numbers.
            """
        )
    st.markdown(
        """
#### What embedding vector looks like?
`torch.nn.functional.embedding()` is basically just picking the corresponding row of the embedding weights for each token ID. It's identical to slicing the embedding weights tensor like `embedding_weights[token_id]`.

Thus, the dimensions of the embedding vector are the same as the second dimension of the embedding weights tensor, which is `1024` for this model.
"""
    )

    st.markdown(
        """
Unfortunately, we can't fully visualize these 1024-dimensional vectors as we live in a 3D world. However, we can still compare the distances between these vectors to see how similar they are to each other. `torch.dot()` calculates the dot product between two vectors, which is a measure of similarity. The higher the dot product, the more similar the vectors are:
"""
    )

    def dot(token1, token2):
        return torch.dot(embedding(tokenize(token1)[0]), embedding(tokenize(token2))[0])

    st.code(inspect.getsource(dot))
    st.code(
        f"""
dot("world", "world")  #=> {dot("world", "world"):.2f} (identical)
dot("world", " world") #=> {dot("world", " world"):.2f}
dot("world", " earth") #=> {dot("world", " earth"):.2f}
dot("world", " math")  #=> {dot("world", " math"):.2f}"""
    )

    with st.expander("_What is dot product?_"):
        st.markdown(
            """
            The dot product is a mathematical operation that takes two equal-length sequences of numbers (vectors) and returns a single number. It is calculated by multiplying corresponding elements of the vectors and summing the results. The dot product can be used to measure the similarity between two vectors: the higher the dot product, the more similar the vectors are.
            """
        )

    st.markdown(
        """
#### Visualizing embedding vectors
Embedding vectors can be visualized in a 2D space using dimensionality reduction techniques.
"""
    )
    fig = create_embedding_clusters(embedding, tokenize)
    st.altair_chart(fig, width="stretch")

    with st.expander("_What is dimensionality reduction techniques?_"):
        st.markdown(
            """
Here, we will use Classical MDS (Multidimensional Scaling) to project the high-dimensional embedding vectors into a 2D space while preserving the pairwise distances as much as possible. This allows us to see how similar tokens cluster together based on their embeddings:
"""
        )

    st.markdown(
        """
As you can see above, each category is clustered well. Also, similar categories like `Cities` and `Countries` are close to each other, while `Colors` are abstract concepts and are therefore far away from other categories.

The very interesting fact here is that this clustering is achieved **without any supervision**. When training the model, we never show these clusters. Instead, we only ask the model to predict the next token and adjust the weights to minimize the prediction error. The next token prediction forces the model to learn the relationships between tokens based on their context in the training data. As a result, the model learns to group similar tokens together in the embedding space.
"""
    )

    # Summary and Progress
    with st.container(border=True):
        st.markdown("**📚 Section Summary**")
        st.markdown(
            "Embedding layers convert token IDs into high-dimensional vectors that capture semantic meaning. Similar tokens cluster together in embedding space through unsupervised learning during next-token prediction training."
        )

        # Progress indicator
        st.progress(3 / 5, text="Step 3/5: What is Embedding? ✅")

    st.divider()

    return embedding


def create_embedding_clusters(embedding, tokenize):
    # Define semantic word clusters for visualization
    cities = ["Paris", "London", "Vancouver"]
    countries = ["France", "England", "Canada"]
    food = ["banana", "bread", "apple", "rice"]
    animals = ["cat", "lion", "bird", "fish"]
    colors = ["red", "blue", "yellow", "purple"]
    words = cities + countries + food + animals + colors

    # Extract embeddings for words that tokenize to single tokens (whitespace-prefixed only)
    selected_embeddings = []
    valid_words = []
    token_ids = []

    for word in words:
        tokens_with_space = tokenize(" " + word)

        if len(tokens_with_space) == 1:
            token_id = tokens_with_space[0]
            selected_embeddings.append(embedding(torch.tensor([token_id])).squeeze(0))  # type: ignore
            valid_words.append("␣" + word)
            token_ids.append(token_id)

    words = valid_words
    selected_embeddings = torch.stack(selected_embeddings)  # type: ignore

    # Apply Classical MDS for dimensionality reduction
    distances = torch.cdist(selected_embeddings, selected_embeddings, p=2)  # type: ignore
    n = distances.shape[0]
    H = torch.eye(n) - torch.ones(n, n) / n
    B = -0.5 * H @ (distances**2) @ H

    eigenvals, eigenvecs = torch.linalg.eigh(B)
    idx = torch.argsort(eigenvals, descending=True)
    eigenvals, eigenvecs = eigenvals[idx], eigenvecs[:, idx]

    embeddings_2d = eigenvecs[:, :2] * torch.sqrt(
        torch.maximum(eigenvals[:2], torch.tensor(0.0))
    )

    # Shift to positive quadrant and scale for better label readability
    x_min, y_min = embeddings_2d.min(dim=0)[0]
    embeddings_2d[:, 0] += abs(x_min) + 0.1 if x_min < 0 else 0
    embeddings_2d[:, 1] += abs(y_min) + 0.1 if y_min < 0 else 0

    # Scale the coordinates to spread out the points more
    scale_factor = 4.0
    embeddings_2d *= scale_factor

    # Add deterministic jitter to prevent exact overlaps (using token IDs as seed)
    jitter_amount = 0.3
    saved = torch.random.get_rng_state()  # Save the current random state
    torch.manual_seed(42)  # Fixed seed for deterministic results
    jitter = torch.randn_like(embeddings_2d) * jitter_amount
    embeddings_2d += jitter

    # Assign categories and create visualization DataFrame
    categories = []
    for word in words:
        # Remove both whitespace symbol and actual whitespace for comparison
        clean_word = word.replace("␣", "").strip()
        if clean_word in cities:
            categories.append("Cities")
        elif clean_word in countries:
            categories.append("Countries")
        elif clean_word in food:
            categories.append("Food")
        elif clean_word in animals:
            categories.append("Animals")
        elif clean_word in colors:
            categories.append("Colors")

    df = pd.DataFrame(
        {
            "x": embeddings_2d[:, 0].detach().numpy(),
            "y": embeddings_2d[:, 1].detach().numpy(),
            "token": words,
            "category": categories,
            "token_id": [
                int(tid.item()) if hasattr(tid, "item") else int(tid)
                for tid in token_ids
            ],
        }
    )

    color_map = {
        "Cities": "red",
        "Countries": "blue",
        "Food": "green",
        "Animals": "orange",
        "Colors": "purple",
    }

    # Create L-shaped coordinate axes with arrows
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()
    origin_x = x_min - 0.1 * (x_max - x_min)
    origin_y = y_min - 0.1 * (y_max - y_min)
    x_end = x_max + 0.05 * (x_max - x_min)
    y_end = y_max + 0.05 * (y_max - y_min)

    # Axis lines
    x_axis = (
        alt.Chart(pd.DataFrame({"x": [origin_x, x_end], "y": [origin_y, origin_y]}))
        .mark_line(color="black", strokeWidth=2)
        .encode(x="x:Q", y="y:Q")
    )
    y_axis = (
        alt.Chart(pd.DataFrame({"x": [origin_x, origin_x], "y": [origin_y, y_end]}))
        .mark_line(color="black", strokeWidth=2)
        .encode(x="x:Q", y="y:Q")
    )

    # Arrow heads
    x_arrow = (
        alt.Chart(pd.DataFrame({"x": [x_end], "y": [origin_y]}))
        .mark_point(shape="triangle-right", size=100, color="black")
        .encode(x="x:Q", y="y:Q")
    )
    y_arrow = (
        alt.Chart(pd.DataFrame({"x": [origin_x], "y": [y_end]}))
        .mark_point(shape="triangle-up", size=100, color="black")
        .encode(x="x:Q", y="y:Q")
    )

    # Scatter plot with circles and data labels
    scatter = (
        alt.Chart(df)
        .mark_circle(size=100, opacity=0.7)
        .encode(
            x=alt.X(
                "x:Q", title="", axis=alt.Axis(labels=False, ticks=False, grid=False)
            ),
            y=alt.Y(
                "y:Q", title="", axis=alt.Axis(labels=False, ticks=False, grid=False)
            ),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(
                    domain=list(color_map.keys()), range=list(color_map.values())
                ),
                legend=alt.Legend(title="Categories", orient="top"),
            ),
            tooltip=alt.value(None),
        )
    )

    # Add data labels offset from the scatter plot
    labels = (
        alt.Chart(df)
        .mark_text(
            align="left",
            baseline="middle",
            fontSize=14,
            fontWeight="normal",
            color="black",
            dx=8,
        )
        .encode(
            x="x:Q",
            y="y:Q",
            text="token:N",
            tooltip=["token:N", "category:N", "token_id:O"],
        )
    )

    # Combine all chart elements
    final_chart = (
        (x_axis + y_axis + x_arrow + y_arrow + scatter + labels)
        .resolve_scale(color="independent")
        .properties(width=600, height=450)
    )

    torch.random.set_rng_state(saved)  # Restore the random state

    return final_chart
