from torch import nn
import torch.nn.functional as F





class CNN3(nn.Module):
   def __init__(self, in_channels, num_classes):

       """
       Building blocks of convolutional neural network.

       Parameters:
           * in_channels: Number of channels in the input image (for grayscale images, 1)
           * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
       """
       super(CNN3, self).__init__()

       # 1st convolutional layer
       self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=5, stride=1, padding=2)
       self.batch1 = nn.BatchNorm2d(24)
       self.relu1 = nn.ReLU()
       self.pool1 = nn.MaxPool2d(kernel_size=2) 


       # 2nd convolutional layer
       self.conv2 = nn.Conv2d(in_channels=24, out_channels=8, kernel_size=5,stride=1, padding=2)
       self.batch2 = nn.BatchNorm2d(8)
       self.relu2 = nn.ReLU()
       self.pool2 = nn.MaxPool2d(kernel_size=2)
       self.fc = nn.Linear(8 * 7 * 7, num_classes)

   def forward(self, x):
       """
       Define the forward pass of the neural network.

       Parameters:
           x: Input tensor.

       Returns:
           torch.Tensor
               The output tensor after passing through the network.
       """
       x = self.conv1(x)
       x = self.batch1(x)
       x = self.relu1(x)

       # Max Pool 1
       x = self.pool1(x)

       # Conv 2
       x = self.conv2(x)
       x = self.batch2(x)
       x = self.relu2(x)

       # Max Pool 2
       x = self.pool2(x)

       x = x.view(x.size(0), -1)
       # Linear Layer
       x = self.fc(x)

       return x




