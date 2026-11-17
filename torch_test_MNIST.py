from cnn import  CNN3
import torch
from  torchmetrics import Accuracy
from  torchmetrics import Precision
from  torchmetrics import Recall
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Create a new model instance
model = CNN3(in_channels=1, num_classes=10)

# Load the saved model
model.load_state_dict(torch.load('MulticlassCNN.pth'))
print(model)


batch_size = 200

test_dataset = datasets.MNIST(root="./data", download=True, train=False, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)







# Set up of multiclass accuracy metric
acc = Accuracy(task="multiclass",num_classes=10)
recall = Recall(task="multiclass",num_classes=10)
precision = Precision(task="multiclass",num_classes=10)


# Iterate over the dataset batches


model.eval()
with torch.no_grad():

   print('len test sample: ', len(test_loader))

   for images, labels in test_loader:
       # Get predicted probabilities for test data batch
       outputs = model(images)
       _, preds = torch.max(outputs, 1)
       acc(preds, labels)
       precision(preds, labels)
       recall(preds, labels)



#Compute total test accuracy
test_accuracy = acc.compute()
print(f"Test accuracy: {test_accuracy}")

