import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
from cnn import  CNN3
import sys


# read a png image from the command line 
image = cv2.imread(sys.argv[1],  cv2.IMREAD_GRAYSCALE)
# resize to the MNIST format
resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

# make the background black
inverted_image = cv2.bitwise_not(resized_image)

# normalize to 0-255
normalized_image = inverted_image



input_tensor = torch.from_numpy(normalized_image).float()     # Convert the numpy array to a PyTorch tensor
input_tensor_pytorch = input_tensor.unsqueeze(0).unsqueeze(0) # Adds batch and channel dimensions



# get the model
model = CNN3(in_channels=1, num_classes=10)
model.load_state_dict(torch.load('MulticlassCNN.pth'))
# disable gradient
with torch.no_grad():
    output = model(input_tensor_pytorch)




predicted_class = torch.argmax(output).item()

print(f"The model predicts the digit is: {predicted_class}")

probabilities = np.round( torch.softmax(output, dim=1).numpy()[0] , decimals=2)

print(f"The probabilities are: {probabilities}")

plt.imshow(normalized_image, cmap='gray' )
plt.show()



