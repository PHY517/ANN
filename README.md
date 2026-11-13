# Training a Convolutional Neural Network to Read Handwritten Digits

This example introduces the basic workflow for training an **artificial neural network (ANN)** with **PyTorch**.

The goal is simple:

> **Train a convolutional neural network (CNN) to recognize handwritten digits from 0 to 9.**

The example uses the **MNIST** handwritten-digit dataset and the `CNN3` architecture defined in `cnn.py`.

---

## 1. The Big Picture

Training a neural network means repeatedly showing the network examples, measuring how wrong its predictions are, and changing its parameters to reduce that error.

```text
MNIST images + labels
        |
        v
     DataLoader
   batches of images
        |
        v
       CNN3
        |
        v
    10 scores
        |
        v
      Loss  <---- correct labels
        |
        v
 Backpropagation
        |
        v
    Gradients
        |
        v
      Adam
        |
        v
Update network parameters
        |
        +-----------------> next batch
```

The essential training cycle is:

```python
scores = model(data)                # 1. Forward pass
loss = criterion(scores, targets)   # 2. Calculate error
optimizer.zero_grad()               # 3. Clear old gradients
loss.backward()                     # 4. Backpropagation
optimizer.step()                    # 5. Update parameters
```

These five operations are the heart of the training program.

---

# 2. Imports

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from cnn import CNN3
```

The main packages have different roles:

| Package | Purpose |
|---|---|
| `torch` | Tensors, automatic differentiation, and general PyTorch functionality |
| `torch.nn` | Neural-network layers and loss functions |
| `torch.optim` | Optimization algorithms such as Adam |
| `torchvision` | Image datasets and image utilities |
| `DataLoader` | Divides a dataset into manageable batches |
| `tqdm` | Displays a progress bar during training |
| `CNN3` | The CNN architecture defined in `cnn.py` |

---

# 3. Choose the Batch Size

```python
batch_size = 200
```

MNIST contains **60,000 training images**. Instead of processing all 60,000 images simultaneously, the network processes groups, or **batches**, of 200 images.

Therefore,

```text
60,000 images / 200 images per batch = 300 batches per epoch
```

A parameter update is performed after every batch.

This means one epoch produces approximately **300 parameter updates**.

---

# 4. Load the MNIST Training Dataset

```python
train_dataset = datasets.MNIST(
    root="./data",
    download=True,
    train=True,
    transform=transforms.ToTensor()
)
```

### `root="./data"`

The MNIST files are stored in the local `./data` directory.

### `download=True`

PyTorch downloads MNIST if it is not already available locally.

### `train=True`

Use the **training portion** of MNIST.

### `transform=transforms.ToTensor()`

Each image is converted into a PyTorch tensor.

An MNIST image has shape

```text
[1, 28, 28]
```

where

```text
1  = grayscale channel
28 = image height
28 = image width
```

`ToTensor()` also converts the original 8-bit pixel intensities into floating-point values approximately between **0 and 1**.

---

# 5. Create the DataLoader

```python
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
```

There are two useful concepts here.

## Dataset

The `Dataset` stores the examples and their labels.

Conceptually:

```text
image 0 -> label 5
image 1 -> label 0
image 2 -> label 4
...
```

## DataLoader

The `DataLoader` retrieves samples from the dataset and organizes them into batches.

One batch has approximately the shapes

```text
data.shape    = [200, 1, 28, 28]
targets.shape = [200]
```

The dimensions of `data` mean

```text
200 = number of images
1   = grayscale channel
28  = height
28  = width
```

Because

```python
shuffle=True
```

the ordering of the training images is randomized rather than presenting them in the same fixed order.

---

# 6. Inspect the Training Data

Before training, the script displays some images.

```python
def imshow(img):
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()
```

A batch is obtained with

```python
dataiter = iter(train_loader)
images, labels = next(dataiter)
```

and displayed as a grid:

```python
imshow(torchvision.utils.make_grid(images))
```

This is **not part of the neural-network training**.

It is a useful sanity check:

- Did the dataset load correctly?
- Are the images actually handwritten digits?
- Are the image dimensions correct?
- Did the preprocessing work?

Inspecting the data before training is an important part of any machine-learning workflow.

---

# 7. Select the Computing Device

The code checks whether a CUDA GPU is available:

```python
print(torch.cuda.is_available())
```

A common portable choice is

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

The example instead uses

```python
device = "mps"
```

`mps` is the PyTorch backend for the Apple Metal Performance Shaders framework.

Typical possibilities are:

| Device | Hardware |
|---|---|
| `cpu` | CPU |
| `cuda` | NVIDIA GPU |
| `mps` | Apple GPU |

A fundamental PyTorch rule is:

> **The model and the data used by the model must be on the same device.**

---

# 8. Create the Neural Network

```python
model = CNN3(
    in_channels=1,
    num_classes=10
).to(device)
```

## Why `in_channels=1`?

MNIST images are grayscale, so each image has one channel.

## Why `num_classes=10`?

The possible output classes are

```text
0 1 2 3 4 5 6 7 8 9
```

so the network needs ten output scores.

At this stage the architecture exists, but the parameters have not yet learned how to recognize digits.

```text
CNN architecture
      +
initial parameters
      |
      v
untrained network
```

Training determines useful values for those parameters.

---

# 9. Define the Loss Function

```python
criterion = nn.CrossEntropyLoss()
```

The **loss function** measures how wrong the network prediction is.

For every input image, `CNN3` produces ten raw output values:

```text
class     0    1    2    3    4    5    6    7    8    9
score    ...  ...  ...  ...  ...  ...  ...  ...  ...  ...
```

These values are commonly called **scores** or **logits**.

`CrossEntropyLoss` compares the scores with the correct class label.

Conceptually:

```text
network scores
       +
correct digit
       |
       v
CrossEntropyLoss
       |
       v
 one loss value
```

A large loss means that the network's predictions disagree strongly with the labels. Training attempts to reduce this loss.

> **Important:** Do not apply `softmax()` before `CrossEntropyLoss`. PyTorch's cross-entropy implementation operates directly on the raw logits.

---

# 10. Define the Optimizer

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

The optimizer determines **how the parameters are changed** once their gradients are known.

The example uses the **Adam** optimizer.

### `model.parameters()`

These are the trainable parameters of the neural network: convolutional kernels, biases, BatchNorm parameters, fully connected weights, and so on.

### `lr=0.001`

The learning rate controls the size of parameter updates.

The conceptual gradient-descent update is

```text
theta_new = theta_old - learning_rate * gradient
```

or mathematically,

```text
θ_new = θ_old - η ∇L
```

where

- `θ` represents all trainable parameters,
- `L` is the loss,
- `∇L` is the loss gradient,
- `η` is the learning rate.

Adam uses a more sophisticated adaptive update, but the physical picture remains the same: **move the parameters in a direction expected to reduce the loss**.

---

# 11. Choose the Number of Epochs

```python
num_epochs = 10
```

An **epoch** is one complete pass through the training dataset.

```text
Epoch 1  -> all 60,000 training images
Epoch 2  -> all 60,000 training images again
...
Epoch 10 -> all 60,000 training images again
```

With 60,000 images and batch size 200:

```text
300 batches / epoch
```

Therefore ten epochs produce approximately

```text
10 x 300 = 3000 optimizer updates
```

---

# 12. Start the Epoch Loop

```python
for epoch in range(num_epochs):
```

The program now makes repeated passes through the dataset.

The current learning rate is printed with

```python
current_lr = optimizer.param_groups[0]['lr']

print(
    f"Epoch [{epoch + 1}/{num_epochs}], "
    f"Learning Rate: {current_lr:.6f}"
)
```

In the current example the learning rate remains `0.001`, but printing it is useful if a learning-rate scheduler is introduced later.

---

# 13. Loop Over the Batches

Inside each epoch:

```python
for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
```

Each iteration provides

```text
data    -> 200 handwritten-digit images
targets -> 200 correct digit labels
```

For example:

```text
data

[ image of 7 ]
[ image of 3 ]
[ image of 8 ]
[ image of 1 ]
      ...


targets

[7, 3, 8, 1, ...]
```

Everything inside this loop represents **one training step**.

---

# 14. Move the Batch to the Computing Device

```python
data = data.to(device)
targets = targets.to(device)
```

If the model is stored on a GPU, the input data and labels must also be placed on that device.

```text
model   ---> device
data    ---> device
targets ---> device
```

The model can then perform the calculations using that hardware.

---

# 15. Step 1: Forward Pass

```python
scores = model(data)
```

The batch of images is propagated through `CNN3`.

Conceptually:

```text
images
   |
   v
convolution
   |
   v
BatchNorm
   |
   v
ReLU
   |
   v
pooling
   |
   v
convolution
   |
   v
BatchNorm
   |
   v
ReLU
   |
   v
pooling
   |
   v
flatten
   |
   v
linear layer
   |
   v
10 output scores
```

For a batch of 200 images:

```text
scores.shape = [200, 10]
```

Each row contains the ten scores for one image.

For example:

```text
Digit:    0     1     2     3     4     5     6     7     8     9
Score:   0.2  -0.8   0.1   1.2   0.4  -0.3   0.5   3.7   0.1  -0.4
                                                ^
                                          largest score
```

The largest score corresponds to digit `7`, so the network would currently classify that image as `7`.

---

# 16. Step 2: Calculate the Loss

```python
loss = criterion(scores, targets)
```

PyTorch compares the network scores with the known correct labels.

```text
predicted scores
       +
correct targets
       |
       v
CrossEntropyLoss
       |
       v
      loss
```

The result is a scalar measuring the error of the current predictions.

Training tries to make this number smaller.

---

# 17. Step 3: Clear Old Gradients

```python
optimizer.zero_grad()
```

PyTorch **accumulates gradients by default**.

Without clearing them, gradients from different batches would be added together:

```text
gradient from batch 1
          +
gradient from batch 2
          |
          v
unwanted accumulated gradient
```

Therefore the gradients from the previous training step are cleared before calculating the new ones.

---

# 18. Step 4: Backpropagation

```python
loss.backward()
```

This is the key automatic-differentiation step.

PyTorch calculates derivatives of the loss with respect to every trainable parameter:

```text
dL/dw1
dL/dw2
dL/db1
...
```

These gradients answer the question:

> **If this parameter changes slightly, how will the loss change?**

Backpropagation efficiently applies the chain rule through all the operations in the neural network.

A very important distinction is:

> `loss.backward()` **calculates gradients. It does not update the model parameters.**

---

# 19. Step 5: Update the Parameters

```python
optimizer.step()
```

Now Adam uses the gradients calculated by

```python
loss.backward()
```

to modify the network parameters.

This is the actual **learning step**.

Conceptually:

```text
old parameters
      +
 gradients
      +
 Adam optimizer
      |
      v
new parameters
```

The new parameters should, on average, move the network toward a lower loss.

---

# 20. The Five Essential Training Operations

The entire neural-network training process can therefore be summarized by five operations:

| Step | Code | Meaning |
|---|---|---|
| 1 | `scores = model(data)` | Make predictions using the current parameters |
| 2 | `loss = criterion(scores, targets)` | Measure the prediction error |
| 3 | `optimizer.zero_grad()` | Remove gradients from the previous batch |
| 4 | `loss.backward()` | Calculate new gradients by backpropagation |
| 5 | `optimizer.step()` | Change the model parameters |

The distinction between the last two steps is particularly important:

```text
loss.backward()
      |
      v
calculate gradients

optimizer.step()
      |
      v
change parameters
```

---

# 21. Repeat Over Batches and Epochs

The five-step cycle is repeated for every batch:

```text
Batch 1
forward -> loss -> backward -> update

Batch 2
forward -> loss -> backward -> update

Batch 3
forward -> loss -> backward -> update

...

Batch 300
forward -> loss -> backward -> update
```

After approximately 300 batches, one epoch is complete.

The process then starts again with the next epoch.

```text
                    EPOCH 1
                       |
        +--------------+--------------+
        |              |              |
     batch 1         batch 2        ... batch 300
        |
        v
                    EPOCH 2
                       |
                      ...
                       |
                    EPOCH 10
```

---

# 22. Save the Trained Model

After training:

```python
torch.save(
    model.state_dict(),
    'MulticlassCNN.pth'
)
```

`model.state_dict()` contains the learned parameter values of the network.

Conceptually:

```text
CNN3 architecture
       +
learned parameters
       |
       v
trained neural network
```

The `.pth` file stores the learned state, but the `CNN3` Python class still defines the architecture.

To use the model later:

```python
model = CNN3(in_channels=1, num_classes=10)
model.load_state_dict(torch.load('MulticlassCNN.pth'))
```

The architecture used when loading the parameters must match the architecture used during training.

---

# 23. Training vs. Testing vs. Inference

It is useful to separate three different stages.

## Training

```text
images + known labels
        |
        v
prediction
        |
        v
loss
        |
        v
gradients
        |
        v
parameter update
```

The model parameters **change**.

## Testing

```text
unseen images + known labels
        |
        v
prediction
        |
        v
compare with labels
        |
        v
accuracy / other metrics
```

The model parameters **do not change**.

Testing measures generalization to data the network did not train on.

## Inference

```text
new image
    |
    v
trained model
    |
    v
predicted digit
```

The correct answer may not be known. The goal is simply to use the trained model to make a prediction.

---

# 24. A Physics Interpretation

For a computational physics course, neural-network training can be viewed as an optimization problem.

The collection of all network parameters can be represented by a high-dimensional coordinate vector

```text
θ = (θ1, θ2, θ3, ...)
```

and the loss is a scalar function

```text
L(θ)
```

Training attempts to find parameters that minimize this function:

```text
minimize L(θ)
```

This is analogous to moving through a high-dimensional potential-energy landscape toward a low-energy configuration.

The gradient

```text
∇L
```

indicates how the loss changes locally in parameter space.

Backpropagation calculates this gradient, and the optimizer uses it to move through parameter space.

```text
parameter space
      |
      v
   L(theta)
      |
      v
calculate gradient
      |
      v
move downhill
      |
      v
lower loss
```

This connects neural-network training directly to numerical minimization methods used throughout computational physics.

---

# 25. A Useful Improvement to the Training Script

Because `CNN3` contains Batch Normalization layers, it is good practice to explicitly place the network in training mode before training:

```python
model.train()
```

For example:

```python
model.train()

for epoch in range(num_epochs):
    for batch_index, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

During testing or inference, switch to

```python
model.eval()
```

This distinction matters for layers such as **BatchNorm** and **Dropout**.

It is also useful to record the average loss for each epoch so students can verify that training is actually progressing.

---

# 26. Take-Home Message

The details of CNNs, cross entropy, Adam, and backpropagation can initially make neural-network training look complicated. The computational workflow, however, is very compact:

```python
for epoch in range(num_epochs):
    for data, targets in train_loader:

        scores = model(data)                # predict
        loss = criterion(scores, targets)   # measure error

        optimizer.zero_grad()               # clear old gradients
        loss.backward()                     # calculate gradients
        optimizer.step()                    # update parameters
```

Remember the sequence:

```text
DATA
 |
 v
MODEL
 |
 v
PREDICTION
 |
 v
LOSS
 |
 v
BACKPROPAGATION
 |
 v
GRADIENTS
 |
 v
OPTIMIZER
 |
 v
NEW PARAMETERS
```

Then repeat.

That repeated cycle is what **training a neural network** means computationally.
