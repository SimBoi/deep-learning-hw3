import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _get_invalid_locations_mask(w, device):
    diagonals_list = []
    # Generate the diagonal masks
    for j in range(-w, 1):
        diagonal_mask = torch.zeros(w, device="cpu", dtype=torch.uint8)
        diagonal_mask[:-j] = 1
        diagonals_list.append(diagonal_mask)

    # Stack the diagonal masks to create the mask
    mask = torch.stack(diagonals_list, dim=-1)

    # Add extra dimensions to the mask
    mask = mask[None, :, None, :]

    # Create the ending mask by flipping the mask
    ending_mask = mask.flip(dims=(1, 3)).bool().to(device)

    return mask.bool().to(device), ending_mask


def mask_invalid_locations(input_tensor: torch.Tensor, w: int):
    # Get the masks for invalid locations
    beginning_mask, ending_mask = _get_invalid_locations_mask(w, input_tensor.device)

    # Get the sequence length
    seq_len = input_tensor.size(1)

    # Mask the beginning of the tensor
    beginning_input = input_tensor[:, :w, :, : w + 1]
    beginning_mask = beginning_mask[:, :seq_len].expand(beginning_input.size())
    beginning_input.masked_fill_(beginning_mask, -9e15)

    # Mask the end of the tensor
    ending_input = input_tensor[:, -w:, :, -(w + 1) :]
    ending_mask = ending_mask[:, -seq_len:].expand(ending_input.size())
    ending_input.masked_fill_(ending_mask, -9e15)


def _skew(x, direction, padding_value):
    x_padded = nn.functional.pad(x, direction, value=padding_value)
    x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
    return x_padded


def get_main_diagonals_indices(b, n, w):
    diag_indices = torch.arange(-w, w + 1)
    row_indices = torch.arange(0, n * n, n + 1)
    col_indices = row_indices.view(1, -1, 1) + diag_indices
    # Repeat the column indices for each batch
    col_indices = col_indices.repeat(b, 1, 1)
    # Flatten the column indices and remove the padding
    col_indices = col_indices.flatten(1)[:, w:-w]
    return col_indices


def populate_diags(x):
    # Get the dimensions
    bzs, seq_len, w_ = x.size()
    # Compute the half window size
    w = (w_ - 1) // 2
    # Flatten the tensor and remove the padding
    x = x.flatten(1)[:, w:-w].float()
    # Initialize the output tensor
    res = torch.zeros(bzs, seq_len, seq_len, device=x.device).flatten(1)
    # Get the indices of the main diagonals
    idx = get_main_diagonals_indices(bzs, seq_len, w).to(x.device)
    # Populate the main diagonals
    res = res.scatter_(1, idx, x).view(bzs, seq_len, seq_len)
    return res


def sliding_chunks_matmul_qk(q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float):
    # Get the dimensions
    bsz, num_heads, seqlen, head_dim = q.size()
    assert q.size() == k.size()
    chunks_count = seqlen // w - 1

    # Reshape the query and key tensors
    q = q.reshape(bsz * num_heads, seqlen, head_dim)
    k = k.reshape(bsz * num_heads, seqlen, head_dim)

    # Unfold the query and key tensors into chunks
    chunk_q = q.unfold(-2, 2*w, w).transpose(-1,-2)
    chunk_k = k.unfold(-2, 2*w, w).transpose(-1,-2)

    # Compute the attention scores
    chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))  # multiply

    # Apply the skewing operation
    diagonal_chunk_attn = _skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)

    # Initialize the diagonal attention tensor
    diagonal_attn = torch.ones((bsz * num_heads, chunks_count + 1, w, w * 2 + 1), device=chunk_attn.device)*(-9e15)

    # Populate the diagonal attention tensor
    diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, :w + 1]
    diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, :w + 1]
    diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, - (w + 1):-1, w + 1:]
    p = w > 1
    diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, :w - 1, p-w:]

    # Reshape and transpose the diagonal attention tensor
    diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1).transpose(2, 1)

    # Mask invalid locations
    mask_invalid_locations(diagonal_attn, w)

    # Reshape and transpose the diagonal attention tensor again
    diagonal_attn = diagonal_attn.transpose(1,2).view(bsz*num_heads, seqlen, 2 * w + 1)

    return diagonal_attn


def _skew2(x, padding_value):
    # Get the dimensions
    B, C, M, L = x.size()
    # Pad the tensor
    x = F.pad(x, (0, M + 1), value=padding_value)
    # Remove the last M elements
    x = x.view(B, C, -1)
    x = x[:, :, :-M]
    # Remove the last element in the last dimension
    x = x.view(B, C, M, M + L)
    x = x[:, :, :, :-1]
    return x


def sliding_chunks_matmul_pv(prob: torch.Tensor, v: torch.Tensor, w: int):
    # Get the dimensions
    bsz, seqlen, num_heads, head_dim = v.size()

    # Ensure the sequence length is a multiple of the window size
    assert seqlen % (w * 2) == 0
    # Ensure the dimensions of the probabilities and value vectors match
    assert prob.size()[:3] == v.size()[:3]
    assert prob.size(3) == 2 * w + 1

    chunks_count = seqlen // w - 1

    # Reshape the probabilities into chunks
    chunk_prob = prob.transpose(1, 2).reshape(
        bsz * num_heads, seqlen // w, w, 2 * w + 1
    )

    # Reshape the value vectors
    v = v.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    # Pad the value vectors
    padded_v = F.pad(v, (0, 0, w, w), value=-1)

    # Compute the size and stride for the chunked value vectors
    chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
    chunk_v_stride = padded_v.stride()
    chunk_v_stride = (
        chunk_v_stride[0],
        w * chunk_v_stride[1],
        chunk_v_stride[1],
        chunk_v_stride[2],
    )

    # Create the chunked value vectors
    chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)
    # Skew the probabilities
    skewed_prob = _skew2(chunk_prob, padding_value=0)
    # Compute the output values
    context = torch.einsum("bcwd,bcdh->bcwh", (skewed_prob, chunk_v))

    return context.view(bsz, num_heads, seqlen, head_dim)


def pad_qkv_to_window_size(q,k,v,one_sided_window_size, padding_mask, padding_value=0):
    # Get the sequence length from the query vectors
    seq_len = q.shape[-2]

    # Compute the total window size
    w = int(2 * one_sided_window_size)

    # Compute the padding length
    padding_len = (w - seq_len % w) % w

    # Compute the left and right padding lengths
    padding_l, padding_r = (padding_len // 2, padding_len // 2) if w > 2 else (0, 1)

    # Pad the query, key, and value vectors
    q = F.pad(q, (0, 0, padding_l, padding_r), value=padding_value)
    k = F.pad(k, (0, 0, padding_l, padding_r), value=padding_value)
    v = F.pad(v, (0, 0, padding_l, padding_r), value=padding_value)

    # If a padding mask is provided, pad it as well
    if padding_mask is not None:
        padding_mask = F.pad(padding_mask, (padding_l, padding_r), value=0)

    return q, k, v, padding_mask


def sliding_window_attention(q, k, v, window_size, padding_mask=None):
    '''
    Computes the simple sliding window attention from 'Longformer: The Long-Document Transformer'.
    This implementation is meant for multihead attention on batched tensors. It should work for both single and multi-head         attention.
    :param q - the query vectors. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param k - the key vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param v - the value vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param window_size - size of sliding window. Must be an even number.
    :param padding_mask - a mask that indicates padding with 0.  #[Batch, SeqLen]
    :return values - the output values. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :return attention - the attention weights. #[Batch, SeqLen, SeqLen] or [Batch, num_heads, SeqLen, SeqLen]
    '''

    assert window_size%2 == 0, "window size must be an even number"
    seq_len = q.shape[-2]
    embed_dim = q.shape[-1]
    batch_size = q.shape[0]
    values, attention = None, None

    # Calculate the half window size
    w = window_size // 2

    # Check if there are no heads
    no_head = False
    if len(q.shape) == 3:
        no_head = True
        # Add an extra dimension for the head
        q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)

    num_heads = q.shape[1]

    # Pad the query, key, value vectors and the padding mask to the window size
    q, k, v, padding_mask = pad_qkv_to_window_size(q, k, v, w, padding_mask)

    # Transpose the value vectors for matmul
    v = v.transpose(1, 2)

    new_seq_len = q.shape[-2]

    # Compute the attention scores
    scores = sliding_chunks_matmul_qk(q, k, w, padding_value=-9e15).view(batch_size, num_heads, new_seq_len, 2 * w + 1)

    # If a padding mask is provided, adjust the attention scores accordingly
    if padding_mask is not None:
        padding_mask = torch.logical_not(padding_mask.unsqueeze(dim=1).unsqueeze(dim=-1))
        padding_mask = padding_mask.type_as(q).masked_fill(padding_mask, -9e15)
        ones = padding_mask.new_ones(size=padding_mask.size())  # tensor of ones
        d_mask = sliding_chunks_matmul_qk(ones, padding_mask, w, padding_value=-9e15).view(batch_size, 1, new_seq_len, 2 * w + 1)
        scores += d_mask  # [batch_size, num_heads, seq_len, 2w+1]

    # Compute the attention weights
    attention = torch.nn.functional.softmax(scores / math.sqrt(embed_dim), dim=-1).transpose(1, 2)

    # set the values of the softmax output to 0 for the padded token
    if padding_mask is not None:
        # set the softmax_mask to False for 0.0 and True for -inf in the d_mask
        softmax_mask = (d_mask == -9e15).transpose(1, 2)
        # expand the mask from shape [batch_size, seq_len, 1, 2w+1] to shape [batch_size, seq_len, num_heads, 2w+1]
        softmax_mask = softmax_mask.expand(batch_size, new_seq_len, num_heads, 2 * w + 1)
        # set the attention values to 0 for the padded tokens
        attention = attention.masked_fill(softmax_mask, 0.0)

    # Compute the output values
    values = sliding_chunks_matmul_pv(attention, v, w)  # [batch_size, num_heads, seq_len, embed_dim]

    # Adjust the shape of the attention weights
    attention = attention.transpose(2, 1).contiguous().view(batch_size * num_heads, new_seq_len, 2 * w + 1)
    attention = populate_diags(attention).view(batch_size, num_heads, new_seq_len, new_seq_len)  # [batch_size, num_heads, seq_len, seq_len]

    # If the sequence length has changed, adjust the attention weights and output values accordingly
    if new_seq_len != seq_len:
        pad = new_seq_len - seq_len
        padding_l, padding_r = (pad // 2, pad // 2) if pad > 1 else (0, 1)
        attention = attention[:, :, padding_l:-padding_r, padding_l:-padding_r]
        values = values[:, :, padding_l:-padding_r, :]

    # If there are no heads, remove the extra dimension
    if no_head:
        values = values.squeeze(1)
        attention = attention.squeeze(1)

    return values, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads, window_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        # Stack all weight matrices 1...h together for efficiency
        # "bias=False" is optional, but for the projection we learned, there is no teoretical justification to use bias
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation of the paper if you would like....
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, padding_mask, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, 3*Dims]

        q, k, v = qkv.chunk(3, dim=-1) #[Batch, Head, SeqLen, Dims]

        # Determine value outputs
        # TODO:
        # call the sliding window attention function you implemented
        # ====== YOUR CODE: ======
        values, attention = sliding_window_attention(q, k, v, self.window_size, padding_mask)
        # ========================

        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim) #concatination of all heads
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, window_size, dropout=0.1):
        '''
        :param embed_dim: the dimensionality of the input and output
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param num_heads: the number of heads in the multi-head attention
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, embed_dim, num_heads, window_size)
        self.feed_forward = PositionWiseFeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        '''
        :param x: the input to the layer of shape [Batch, SeqLen, Dims]
        :param padding_mask: the padding mask of shape [Batch, SeqLen]
        :return: the output of the layer of shape [Batch, SeqLen, Dims]
        '''
        # TODO:
        #   To implement the encoder layer, do the following:
        #   1) Apply attention to the input x, and then apply dropout.
        #   2) Add a residual connection from the original input and normalize.
        #   3) Apply a feed-forward layer to the output of step 2, and then apply dropout again.
        #   4) Add a second residual connection and normalize again.
        # ====== YOUR CODE: ======

        # Apply attention to the input x, and then apply dropout.
        attn_output = self.dropout(self.self_attn(x, padding_mask))

        # Add a residual connection from the original input and normalize.
        x = self.norm1(x + attn_output)

        # Apply a feed-forward layer to the output of step 2, and then apply dropout again.
        ff_output = self.dropout(self.feed_forward(x))

        # Add a second residual connection and normalize again.
        x = self.norm2(x + ff_output)

        # ========================

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_length, window_size, dropout=0.1):
        '''
        :param vocab_size: the size of the vocabulary
        :param embed_dim: the dimensionality of the embeddings and the model
        :param num_heads: the number of heads in the multi-head attention
        :param num_layers: the number of layers in the encoder
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param max_seq_length: the maximum length of a sequence
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability

        '''
        super(Encoder, self).__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, hidden_dim, num_heads, window_size, dropout) for _ in range(num_layers)])

        self.classification_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the logits  [Batch]
        '''
        output = None

        # TODO:
        #  Implement the forward pass of the encoder.
        #  1) Apply the embedding layer to the input.
        #  2) Apply positional encoding to the output of step 1.
        #  3) Apply a dropout layer to the output of the positional encoding.
        #  4) Apply the specified number of encoder layers.
        #  5) Apply the classification MLP to the output vector corresponding to the special token [CLS]
        #     (always the first token) to receive the logits.
        # ====== YOUR CODE: ======

        # Apply the embedding layer to the input
        x = self.encoder_embedding(sentence)

        # Apply positional encoding to the output of step 1
        x = self.positional_encoding(x)

        # Apply a dropout layer to the output of the positional encoding
        x = self.dropout(x)

        # Apply the specified number of encoder layers
        for layer in self.encoder_layers:
            x = layer(x, padding_mask)

        # Apply the classification MLP to the output vector corresponding to the special token [CLS]
        cls_token = x[:, 0, :]
        output = self.classification_mlp(cls_token).squeeze(-1)

        # ========================


        return output

    def predict(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the binary predictions  [Batch]
        '''
        logits = self.forward(sentence, padding_mask)
        preds = torch.round(torch.sigmoid(logits))
        return preds
