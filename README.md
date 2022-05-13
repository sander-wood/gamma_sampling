Samplings
=========

A collection of sampling methods for machine learning implemented on numpy.

You can install the python package `samplings` via `pip install samplings`.

For more information, see our paper: [arXiv paper](https://arxiv.org/pdf/2205.06036.pdf).

Author: Sander Wood

Provides
  1. Gamma sampling for controllable generation
  2. Local temperature sampling with weights
  3. Nucleus sampling (top-p sampling)
  4. Top-k sampling
  5. Random sampling

How to use
----------
As you probably know, sampling means randomly picking the next token according 
to its conditional probability distribution. In other words, given the same 
probability distribution, you may get a different result each time. If you 
want to get rid of this uncertainty, you can set `seed` to a fixed value.

By default, all the functions in `samplings` return the index of the next token.
However, you can ask them to return the modified probability distribution by set 
`return_probs` as `True`. Then, you can make further manipulations based on this 
probability distribution.

In addition to the probability distribution, most sampling methods require other
parameters for modifying the probability distribution to achieve desired results.
Please refer to the docstring of each function for details.

About gamma sampling
----------------------
Gamma sampling is a method of tuning probabilities of selected tokens to achieve 
controlling specific properties of the generated sequence. The basic assumption is 
that some attributes of sequences are closely related to the frequencies of some tokens. 
As long as the controllable attribute can be defined at the token level, prior knowledge 
can be directly brought into the sampling process, allowing arbitrary models to support 
customisable and controllable generation.
