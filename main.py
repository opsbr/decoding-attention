from functools import cache, partial
from typing import cast

import torch
from torch import Tensor, nn
from jaxtyping import Int64, Float
from beartype import beartype
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(text: str) -> Int64[Tensor, "seq_len"]:
    return tokenizer(text, return_tensors="pt")["input_ids"][0]


Hello_World = tokenize("Hello World")
print(Hello_World)
Learn_Transformer = tokenize("Learn Transformer")
print(Learn_Transformer)
EndToken = tokenize("<|endoftext|>")
print(EndToken)


model = nn.Linear(1, tokenizer.vocab_size, bias=False)


def train_manually(vocab_size: int) -> Float[Tensor, "{vocab_size} 1"]:
    weight = torch.zeros(vocab_size, 1)
    next_tokens = [
        Hello_World[1],
        Learn_Transformer[1],
        EndToken[0],
    ]
    for next_token in next_tokens:
        weight[next_token - 1] = 1.0
    return weight


model.load_state_dict({"weight": train_manually(tokenizer.vocab_size)})
print(model.weight)

y = torch.softmax(model(x.float()), dim=-1)
print(y)
print(torch.multinomial(y, num_samples=1))
