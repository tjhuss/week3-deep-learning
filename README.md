# Deep Learning for Text and Sequential Data

Week 3 of a self-directed AI/ML/DL internship prep program. Unlike Week 2's
classical ML work, this week builds up deep learning from the ground up --
starting with a single neuron built by hand in plain NumPy, before
introducing PyTorch/TensorFlow, activation functions, and eventually
RNN/LSTM/GRU models for text.

## Setup

```
pip install -r requirements.txt
```

## Project structure

| Folder | Contents |
| --- | --- |
| `day1/` | `day1_neuron.ipynb` -- a single neuron built from scratch with NumPy: forward pass, MSE loss, manually-derived gradients, and a gradient descent training loop |

## Day 1: Deep Learning Fundamentals

`day1/day1_neuron.ipynb` builds one neuron with no framework and no
activation function on purpose -- the goal was the underlying mechanics,
not a real model. It learns `y = 2x + 1` from 5 clean data points, starting
both the weight and bias at 0 so any correct prediction afterward can only
have come from actual training.

Over 1000 epochs of gradient descent, the weight converged to 2.0039 and
the bias to 0.9860 (true values: 2 and 1), with the loss dropping from
57.0 down to essentially 0. The loss curve shows the classic shape: a
sharp drop in the first ~20 epochs, then flattening out as the parameters
get close to correct and each update only needs to fine-tune slightly.
