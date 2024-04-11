r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======

    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======

    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

"""

part1_q2 = r"""
**Your answer:**

"""

part1_q3 = r"""
**Your answer:**

"""

part1_q4 = r"""
**Your answer:**


"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.1
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.5, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**


"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0,
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers['embed_dim'] = 128
    hypers['num_heads'] = 8
    hypers['num_layers'] = 2
    hypers['hidden_dim'] = 512
    hypers['window_size'] = 128
    hypers['droupout'] = 0.22
    hypers['lr'] = 0.000549
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
Stacking encoder layers with sliding-window attention in a transformer model allows each subsequent layer to capture a wider context from the sequence, as each layer’s attention is informed by the context processed by the layer below it. This is similar to how stacking convolutional layers in a CNN enables higher layers to perceive larger receptive fields, thus capturing broader features from the input.
"""

part3_q2 = r"""
**Your answer:**
Dilated sliding-window attention can be designed to skip inputs at a regular interval within each window, effectively expanding the scope of attention while keeping the computational complexity on par with standard sliding-window attention. This method allows the model to integrate information from a wider range of the sequence without a proportional increase in computation, akin to dilated convolutions in image processing.
"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**
Fine-tuning the internal layers, such as the multi-headed attention block, could potentially allow the model to adapt to the task, as these layers are responsible for identifying key patterns in the data. However, the performance might not be as good as when fine-tuning the final layers. This is because the final layers are typically more tailored to the specific task, while the internal layers capture more general language features, which are likely beneficial across a variety of tasks.

"""


# ==============
