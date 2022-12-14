import torch
from torch import nn

import gradio as gr

import requests
from PIL import Image

import ast
import json


import numpy as np
import torch.nn.functional as F

class Net(nn.Module):
    """Custom network predicting the next character of a string.

    Parameters
    ----------
    vocab_size : int
        The number of characters in the vocabulary.

    embedding_dim : int
        Dimension of the character embedding vectors.
    
    dense_dim : int
        Number of neurons in the linear layer that follows the LSTM.

    hidden_dim : int
        Size of the LSTM hidden state.    

    max_norm : int
        If any of the embedding vectors has a higher L2 norm than `max_norm`
        it is rescaled.

    n_layers : int
        Number of the layers of the LSTM.
    """
    def __init__(
        self,
        vocab_size,
        embedding_dim=2,
        norm_type=2,
        max_norm=2,
        window_size=3,
        dense_dim=32,
        hidden_dim=8,
        n_layers=1
    ):
        super().__init__()

        self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=vocab_size - 1,
                norm_type=2,
                max_norm=max_norm,
        )
        self.lstm = nn.LSTM(
                embedding_dim, hidden_dim, batch_first=True, num_layers=n_layers
        )
        
        self.linear_1 = nn.Linear(hidden_dim, dense_dim)
        self.linear_2 = nn.Linear(dense_dim, vocab_size)


    def forward(self, x, h=None, c=None):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(n_samples, window_size)` of dtype
            `torch.int64`.

        h, c : torch.Tensor or None
            Hidden states of the LSTM.

        Returns
        -------
        logits : torch.Tensor
            Tensor of shape `(n_samples, vocab_size)`.

        h, c : torch.Tensor or None
            Hidden states of the LSTM.
        """
        emb = self.embedding(x)  # (n_samples, window_size, embedding_dim)
        if h is not None and c is not None:
            _, (h, c) = self.lstm(emb, (h, c))
        else:
            _, (h, c) = self.lstm(emb)  # (n_layers, n_samples, hidden_dim)

        h_mean = h.mean(dim=0)  # (n_samples, hidden_dim)
        x = self.linear_1(h_mean)  # (n_samples, dense_dim)
        logits = self.linear_2(x)  # (n_samples, vocab_size)

        return logits, h, c

with open("setup_config.json") as f:
    setup_config = json.load(f)

with open("ch2ix.json") as f:
    ch2ix = json.load(f)

with open("vocab.json") as f:
    vocab = json.load(f)

with open("code_to_text.json") as f:
    code_to_text = json.load(f)

with open("text_to_code.json") as f:
    text_to_code = json.load(f)

window_size = int(setup_config["window_size"])
vocab_size = int(setup_config["vocab_size"])
embedding_dim = int(setup_config["embedding_dim"])
norm_type = int(setup_config["norm_type"])
max_norm = int(setup_config["max_norm"])

model = Net( vocab_size, embedding_dim, norm_type, max_norm, window_size)
model.load_state_dict(torch.load("model_lstm.pth"))
model.eval()


def process_text(text):
    raw_text = text.split(',')
    initial_text = [text_to_code[item] for item in raw_text if item in text_to_code]
    window_size = int(setup_config["window_size"])
    res = []
    
    for i in range(5):
        initial_text = initial_text[-window_size:]
        features = torch.LongTensor([[ch2ix[c] if c in ch2ix else (int(setup_config["vocab_size"]) - 1) for c in initial_text]])
        logits, h, c = model(features)
        probs = F.softmax(logits[0], dim=0).detach().numpy()
        new_ch = np.random.choice(vocab, p=probs)
        initial_text.append(new_ch)
        res.append(new_ch)
    res = [code_to_text[item] for item in res if item in code_to_text]
    return res

title = "Interactive demo: LSTM based item recomendation"
description = "LSTM based item recomendation."
article = "<p style='text-align: center'></p>"

iface = gr.Interface(fn=process_text, 
                     inputs=gr.Textbox(placeholder="Audio-Visual, Electrical, Heating, Interior Shutters, Low Voltage"), 
                     outputs="text",
                     title=title,
                     description=description,
                     article=article
                     )
iface.launch(debug=False)
