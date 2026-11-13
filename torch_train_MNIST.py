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

# example from https://www.datacamp.com/tutorial/pytorch-cnn-tutorial
# Loading and preprocessing the dataset
# need:  python3.11 -m pip install --upgrade certifi
#        export SSL_CERT_FILE=$(python3.11 -m certifi) 


batch_size = 200

# PyTorch provides two data primitives: torch.utils.data.DataLoader and 
# torch.utils.data.Dataset that allow you to use pre-loaded datasets as 
# well as your own data. Dataset stores the samples and their corresponding 
# labels, and DataLoader wraps an iterable around the Dataset to enable easy 
# access to the samples.

train_dataset = datasets.MNIST(root="./data", download=True, train=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



# check some images

def imshow(img):
   plt.imshow(np.transpose( img.numpy()   , (1, 2, 0)))
   plt.show()

# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))



# define the convolutional neural network arquitecture

print(torch.cuda.is_available())

#device = "cuda" if torch.cuda.is_available() else "cpu"

device = 'mps'


model = CNN3(in_channels=1, num_classes=10).to(device)
print(model)



#####################
# train the network
#####################

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


img_grid = torchvision.utils.make_grid(images)

num_epochs=10
for epoch in range(num_epochs):
 # Iterate over training batches

   current_lr = optimizer.param_groups[0]['lr']
   print(f"Epoch [{epoch + 1}/{num_epochs}],  Learning Rate: {current_lr:.6f}")
   for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
       data = data.to(device)
       targets = targets.to(device)
       scores = model(data)
       loss = criterion(scores, targets)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

# Save the model
torch.save(model.state_dict(), 'MulticlassCNN.pth')


# export PYTORCH_ENABLE_MPS_FALLBACK=1


