import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    all_chars = set(text)
    all_chars = sorted(all_chars)
    indices = list(range(len(all_chars)))

    char_to_idx = dict(zip(all_chars, indices))
    idx_to_char = dict(zip(indices, all_chars))
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    n_removed = 0
    text_clean = text
    for char in text_clean:
        if char in chars_to_remove:
            text_clean.replace(char, '')
            n_removed += 1
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    result = torch.zeros(len(text), len(char_to_idx), dtype=torch.int8)
    for i, char in enumerate(text):
        ind = char_to_idx[char]
        result[i, ind] = 1
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    indexs = torch.argmax(embedded_text, dim=1)
    result = ''.join(idx_to_char[idx.item()] for idx in indexs)
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    total_length = len(text)
    N = (total_length - 1) // seq_len
    actual_length = N * seq_len
    V = len(char_to_idx)

    embedded_text = chars_to_onehot(text[:actual_length + 1], char_to_idx).to(device)
    embedded_samples = embedded_text[:-1]
    embedded_labels = embedded_text[1:]

    samples = torch.reshape(embedded_samples, (N, seq_len, V)).to(device)

    labels = embedded_labels.argmax(dim=1)
    labels = torch.reshape(labels, (N, seq_len)).to(device)
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    result = torch.nn.functional.softmax(y * (1.0 / temperature), dim=dim)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======
    with torch.no_grad():
        my_batch = torch.unsqueeze(chars_to_onehot(start_sequence, char_to_idx), dim=0).to(device, dtype=torch.float)
        x0 = my_batch
        state = None
        while len(out_text) < n_chars:
            y, state = model(x0, state)
            ch_index = torch.multinomial(hot_softmax(y[0, -1, :], temperature=T), 1)[0].item()
            pred_char = idx_to_char[ch_index]
            out_text += pred_char
            x0 = torch.unsqueeze(chars_to_onehot(pred_char, char_to_idx), dim=0).to(device, dtype=torch.float)
    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        idx = []
        num_of_batches = len(self.dataset) // self.batch_size
        for i in range(num_of_batches):
            for j in range(self.batch_size):
                idx.append((j)*(num_of_batches-1) + j + i)
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: READ THIS SECTION!!

        # ====== YOUR CODE: ======
        for i in range(n_layers):
            x_dim = in_dim if i == 0 else h_dim

            params = dict(
                wxz=nn.Linear(x_dim, h_dim, bias=False),
                whz=nn.Linear(h_dim, h_dim, bias=True),
                wxr=nn.Linear(x_dim, h_dim, bias=False),
                whr=nn.Linear(h_dim, h_dim, bias=True),
                wxg=nn.Linear(x_dim, h_dim, bias=False),
                whg=nn.Linear(h_dim, h_dim, bias=True),
            )

            for k, v in params.items():
                self.add_module(f"{k}_{i}", v)

            self.layer_params.append(params)

        self.dropout = nn.Dropout(p=dropout)
        self.why = nn.Linear(h_dim, out_dim, bias=True)
        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO: READ THIS SECTION!!
        # ====== YOUR CODE: ======
        # Loop over layers of the model
        for layer_idx in range(self.n_layers):
            wxz = self.layer_params[layer_idx]["wxz"]
            whz = self.layer_params[layer_idx]["whz"]
            wxr = self.layer_params[layer_idx]["wxr"]
            whr = self.layer_params[layer_idx]["whr"]
            wxg = self.layer_params[layer_idx]["wxg"]
            whg = self.layer_params[layer_idx]["whg"]

            h_t = layer_states[layer_idx]  # (B, H)
            layer_outputs = []

            # Loop over items in the sequence
            for seq_idx in range(seq_len):
                x_t = layer_input[:, seq_idx, :]  # (B, V) or (B, H)

                z_t = torch.sigmoid(wxz(x_t) + whz(h_t))
                r_t = torch.sigmoid(wxr(x_t) + whr(h_t))
                g_t = torch.tanh(wxg(x_t) + whg(r_t * h_t))
                h_t = z_t * h_t + (1 - z_t) * g_t

                layer_outputs.append(h_t)  # (B, H)

            # Save last state as layer state
            layer_states[layer_idx] = h_t

            # Combine the output from each element in the sequence to a tensor
            # representing the output of the entire sequence
            layer_output = torch.stack(layer_outputs, dim=1)  # (B, S, H)

            # Prepare input for next layer
            layer_input = self.dropout(layer_output)  # (B, S, H)

        # Final output: transform the input to the next (non-existent) layer
        layer_output = self.why(layer_input)  # (B, S, O)
        hidden_state = torch.stack(layer_states, dim=1)  # (B, L, H)
        # ========================
        return layer_output, hidden_state
