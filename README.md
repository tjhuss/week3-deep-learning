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
| `day2/` | `day2_activations.ipynb` -- Sigmoid/Tanh/ReLU/Softmax implemented and visualized by hand, a small 2-layer network learning `y = x^2`, and plain gradient descent vs. Adam compared on the same network |

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

## Day 2: Activation Functions and Optimizers

`day2/day2_activations.ipynb` starts by implementing Sigmoid, Tanh, ReLU,
and Softmax by hand and plotting each one, then proves why they matter:
a linear neuron (Day 1) can only ever learn a straight line, so this
notebook builds a small 2-layer network -- one hidden layer of 6 ReLU
neurons -- and teaches it `y = x^2`, a curve a linear neuron structurally
cannot represent.

Trained with plain gradient descent over 2000 epochs, loss dropped from
15.57 to 0.065. The fitted curve is made of visible straight-line segments
rather than a smooth parabola (ReLU's signature), with a flattened bottom
near x=0 where 6 hidden neurons don't provide enough "kink points" to
match the true curve's sharp minimum -- a direct, visible demonstration of
network capacity.

The same network was then reset to its original weights and retrained
with the Adam optimizer instead of plain gradient descent. Adam converged
faster (down to 0.0634 by epoch 1000, barely moving after) and finished
slightly lower overall (0.0565 vs. 0.0650), though both leveled off near
the same floor -- a reminder that the optimizer speeds up training, but
can't exceed the model's own capacity limit.
