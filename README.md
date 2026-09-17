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
| `day3/` | `day3_ann.ipynb` -- a full 3-layer ANN (batchnorm, dropout, early stopping, Adam) built from scratch, trained on the real Week 2 news dataset and compared against the classical ML results |
| `day4/` | `day4_pytorch.ipynb` -- the same network rebuilt in PyTorch (Dataset/DataLoader, nn.Module, autograd, saved weights) instead of hand-written NumPy |
| `day5/` | `day5_tokenization.py` -- tokenizes, builds a vocabulary for, encodes, and pads the news headlines; saves the result as the actual tokenized dataset |
| `day6/` | `day6_lstm.py` -- an Embedding + LSTM text classifier built in PyTorch on Day 5's tokenized sequences, saved weights, this closes out Week 3 |
| `data/` | `news_dataset.csv` -- the shared, growing news dataset used from Day 5 onward (Days 1-4 each used their own fixed snapshot, see below) |
| `scripts/` | `grow_dataset.py` -- re-scrapes all 4 sources and merges any new headlines into `data/news_dataset.csv`, for growing the dataset further in future sessions |

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

## Day 3: ANN for Structured/Text Data

`day3/day3_ann.ipynb` is the real one -- a 3-layer network (649 input
features -> 32 hidden with batchnorm+relu+dropout -> 16 hidden with relu
-> 6 output classes with softmax) trained on the actual Week 2 news
dataset, not a toy example. New stuff in this one: batch normalization,
dropout, early stopping, and cross-entropy loss, all still hand-coded with
NumPy, no framework yet.

Test accuracy came out to 39%, which is worse than every classical model
from Week 2 (Decision Tree and Gradient Boosting both hit 70%). Not a bug
-- the network has around 21,500 learnable parameters but only 70 actual
training rows after splitting off validation data, which is way too much
model for way too little data. Training loss dropped to nearly 0 (fully
memorized the training set) while validation loss got worse the longer it
trained -- classic overfitting, confirmed by testing it again with 5x more
patience for early stopping and getting the exact same result. This ties
back to two things from Week 2 (XGBoost losing to plain Gradient Boosting,
`RandomizedSearchCV` picking a worse model than `GridSearchCV`): more
advanced tooling doesn't automatically win, especially on a dataset this
small.

## Day 4: PyTorch Training Workflow

`day4/day4_pytorch.ipynb` rebuilds the exact same architecture from Day 3,
except with real PyTorch layers (`nn.Linear`, `nn.BatchNorm1d`, etc.)
instead of hand-written weight matrices, `Dataset`/`DataLoader` for
mini-batch training instead of full-batch, and autograd (`loss.backward()`,
`optimizer.step()`) instead of a hand-derived backward pass. Trained
weights get saved to `news_classifier_weights.pth`.

Test accuracy came out to 52%, better than Day 3's from-scratch result
(39%) but still below every classical model from Week 2 (61-70%). Best
guess for the improvement: this run trains on all 88 rows directly (no
validation split carved out) and uses mini-batches instead of one
full-batch update per epoch, so more individual gradient updates happen
per epoch. Training loss still collapsed to near 0 though -- same
overfitting story as Day 3, framework or not. Also hit a real PyTorch
gotcha here: forgetting `torch.manual_seed(42)` meant every run gave a
different accuracy (52%, then 43%, same code) since PyTorch has its own
separate random number generator that NumPy's seed doesn't control.

## Day 5: Text Representation for Deep Learning

`day5/day5_tokenization.py` switches from bag-of-words (which throws away
word order) to representing each headline as an ordered sequence of words
-- what RNN/LSTM/GRU (Day 6) actually need. Every headline gets lowercased
and split into words, a vocabulary is built from the training set only
(1417 unique words), each headline is converted into a list of vocabulary
numbers (with `0` reserved for both padding and unknown/out-of-vocabulary
words -- a simplification worth knowing about, not a fully correct setup),
and every sequence is padded or truncated to a fixed length of 20 (based
on headline lengths ranging 2-24 words). The final tokenized dataset gets
saved to disk (`train_padded.npy`, `test_padded.npy`, label CSVs, and
`vocab.json`) as the actual deliverable, ready for Day 6.

## Day 6: RNN, LSTM, GRU, and BiLSTM

`day6/day6_lstm.py` closes out Week 3 with an actual sequence model --
built on Day 5's tokenized/padded headlines instead of bag-of-words. Two
new PyTorch pieces: `nn.Embedding` (turns each word index into a learnable
dense 32-number vector, instead of one giant sparse vector per headline)
and `nn.LSTM` (processes the sequence of word vectors one at a time, in
order, carrying a hidden state forward as memory -- unlike the ANN, which
saw a whole headline at once with no sense of word order).

Test accuracy came out to 50%, close to Day 4's ANN (52%) and well above
Day 3's from-scratch version (39%), but still below every classical model
from Week 2 (61-70%). Expected going in -- RNN-family models are generally
*more* data-hungry than plain ANNs, not less, so getting roughly comparable
results to Day 4 rather than notably worse is a reasonable outcome given
only 222 training rows. This wraps up Week 3's overall deliverable
(text-based DL project using ANN, RNN, LSTM, GRU, or BiLSTM).

## Growing the dataset (before Day 5)

Days 1-4 all used the original 111-row news dataset from Week 2, and
those notebooks are left exactly as they were built. Starting with Day 5,
a bigger, shared dataset lives in `data/news_dataset.csv` instead.

`scripts/grow_dataset.py` re-scrapes fool.com, tradingview.com, and
marketscreener.com, plus finance.yahoo.com's healthcare page and
apnews.com's politics page (both added to fix thin categories), and
oilprice.com for energy. Run it again any time from inside `scripts/`
to pull in whatever's new.

Current totals: **843 rows** (Business 363, Markets 205, Technology
119, Politics 56, Energy 56, Health 44).
