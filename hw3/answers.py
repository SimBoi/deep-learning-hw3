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
    hypers['batch_size'] = 400
    hypers['seq_len'] = 128
    hypers['h_dim'] = 128
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.25
    hypers['learn_rate'] = 0.002
    hypers['lr_sched_factor'] = 0.1
    hypers['lr_sched_patience'] = 2
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "He who knows"
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""

There are two reasons to split the corpus, first due to the text being too long and not having enough memory to train on its entirety.
Second is numerical stability when it comes to the gradients.

"""

part1_q2 = r"""

Hidden states or history from previous samples can affect the output of current ones, which is why they can also take extra memory that show in the text.

"""

part1_q3 = r"""

The order of batches being fed to the model is important to achieve the wanted result.
RNN is a sequential model where the history while feeding each batch to the model can affect the current output, so shuffling the batches might change the training result.

"""

part1_q4 = r"""

1. The lower the temperature the less randomized the characters, so when we train the model it's preferable to expose the model
   to a wider variety of characters and words and set the Tempertaure to high values, but when testing we want to check with gramatically correct
   words so we lower the temperature to 1 to make it more strict.

2. As we saw in the softmax run; setting very high temperature values leads to near uniform distribution, so sampling from it will be very random.

3. As opposed to the case above, very low temperature values lead to a very strict and way lower variety, meaning sampling will yield more predictable results.

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
    hypers['batch_size'] = 10
    hypers['h_dim'] = 256
    hypers['z_dim'] = 32
    hypers['x_sigma2'] = 1e-3
    hypers['learn_rate'] = 0.001
    hypers['betas'] = (0.9, 0.999)
    # ========================
    return hypers


part2_q1 = r"""

x_sigma2 describes the variance of the likelihood distribution, or in other words, the uncertainty of our model.
lower values result in more varied generated images, while high values give more limited and less varied images.

"""

part2_q2 = r"""

1. Reconstruction loss measures how much data is lost when reconstructing data, or how well the model reconstructs data, which means how close the output is to the original input.
   KL divergence loss measures how much data is lost when using posterios distribution of points in latent space to represent the prior distribution, or how close both distributions to each other.

2. The lower the KL divergence loss the less the mean and variance of the distance between the posterior distribution in latent space and the prior distribution, which helps us generate images close to the originals.

3. It helps us generate images close to the ones in the dataset, and generate good new coherent data.

"""

part2_q3 = r"""

Our goal is to be able to create a probabilistic model that represents a prediction for the model our samples came from.
This in a sense, means that we want to maximize the evidence probability, or the distribution of the instance space due to the generative process.

"""

part2_q4 = r"""

It is more numerically stable this way, variances tend to be very small, so modeling them directly forces us to deal with small numbers that leaves us prone to numerical error, using the log function to project this number solves the problem.

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
