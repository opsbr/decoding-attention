import streamlit as st
import pandas as pd  # type: ignore
import inspect


def section2(pretrained):
    return pretrained.state_dict()


def run(pretrained):
    st.markdown(
        """
### Section 2: Pretrained weights
In this section, let's check the pretrained weights of the model. Weights are the learned parameters of the model from training. They are the numbers that are used when the model calculates logits. By inspecting the weights here, you'll get a better understanding of how the model works internally.
"""
    )
    st.code(inspect.getsource(section2), line_numbers=True)
    weights = section2(pretrained)

    st.markdown(
        """
#### Pretrained weights
In PyTorch models, weights are stored in a dictionary format and can be accessed using the `state_dict()` method. The keys in this dictionary represent different parts of the model. Let's take a look at all the keys of the pretrained model weights and their shapes:
"""
    )
    df = create_weights_table(weights)
    st.dataframe(df)

    st.markdown(
        """
At this moment, you don't need to understand what each key means. Just know that these keys represent the set of numbers that the model uses to calculate logits. For this model, all the weights' shapes are either 1D or 2D arrays (tensors), and the number of parameters is the number of elements in each array/tensor.
"""
    )

    st.markdown(
        """
#### Total number of parameters (i.e. model size)
To see the total parameters of this model, often called the model size, we can sum the number of parameters of all the weights as shown below:
"""
    )
    st.code(
        f"""
sum(
    tensor.numel()
    for key, tensor in weights.items()
    if "lm_head" not in key
)"""
    )
    num_params = sum(
        tensor.numel() for key, tensor in weights.items() if "lm_head" not in key
    )
    st.markdown(f"**The model size:** `{num_params:,}` (= ~0.6B)")
    st.markdown(
        """
As you can see, the total number of parameters in this model is ~0.6B, which is implied by the model name `Qwen/Qwen3-0.6B`.
"""
    )
    st.info(
        """
Note: The `lm_head` is not included in this count because it is tied to the `embed_tokens` weights, which means they share the same parameters under the hood. This is a common technique used to reduce the number of parameters in language models.
"""
    )

    st.markdown(
        """
#### Objectives
The goal of the rest of "Decoding Attention" is **to build the model architecture that can calculate logits using these weights**. As long as we build the same architecture as the pretrained model, the calculation results will be identical. Throughout this activity, you'll learn about each part of the model architecture and how it works.

"""
    )

    # Summary and Progress
    with st.container(border=True):
        st.markdown("**📚 Section Summary**")
        st.markdown(
            "We explored the pretrained model weights stored as PyTorch state_dict, understanding that these ~0.6B parameters define the model's learned knowledge and will be used to rebuild the architecture."
        )

        # Progress indicator
        st.progress(2 / 5, text="Step 2/5: Pretrained Weights ✅")

    st.divider()

    return weights


def create_weights_table(weights):
    # Build table data
    table_data = []

    for key in weights.keys():
        shape = weights[key].shape
        num_params = weights[key].numel()

        # Add note for lm_head about weight tying
        key_display = key
        if "lm_head" in key:
            key_display = f"{key} (shared with embed_tokens)"

        table_data.append(
            {
                "Key": key_display,
                "Shape": str(tuple(shape)),
                "# of Parameters": f"{num_params:,}",
            }
        )

    return pd.DataFrame(table_data)
