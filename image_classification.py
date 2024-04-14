
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os


# In[4]:


# Define data transformations for data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# In[5]:


# Define the data directory
data_dir = "E:\COVID-19 DETECTION\Classification using ResNet-18\dataset"

# Create data loaders
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
#image_datasets


# In[6]:


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)

class_names = image_datasets['train'].classes
class_names


# In[7]:


# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Freeze all layers except the final classification layer
for name, param in model.named_parameters():
    if "fc" in name:  # Unfreeze the final classification layer
        param.requires_grad = True
    else:
        param.requires_grad = False

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Use all parameters


# Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# In[8]:


# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

print("Training complete!")


# In[9]:



# Save the model
torch.save(model.state_dict(), 'covid_classification_model2.pth')


# # Classification on Unseen Image

# To use the saved model to classify unseen images, you need to load the model and then apply it to the new images for inference. 

# In[8]:


import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load a pre-trained ResNet-18 model with the desired output units
new_model = models.resnet18(pretrained=True)
new_model.fc = nn.Linear(new_model.fc.in_features, 2)  # Adjust to match the desired output units

# Load and preprocess the unseen image
image_path = "E:/COVID-19 DETECTION/Classification using ResNet-18/normal chest image x-ray.jpg"  # Replace with the path to your image
image = Image.open(image_path)

# Convert the image to RGB if it's grayscale
if image.mode != 'RGB':
    image = image.convert('RGB')

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to a smaller resolution
    transforms.ToTensor(),
])

input_tensor = preprocess(image)  # Convert image to tensor
input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = new_model(input_tensor)

# Get the predicted class
_, predicted_class = output.max(1)

# Map the predicted class to the class name
class_names = ['COVID', 'Normal']  # Make sure these class names match your training data
predicted_class_name = class_names[predicted_class.item()]

print(f'The predicted class is: {predicted_class_name}')




# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image

# # Load the saved model
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust to match the original model's output units
# model.load_state_dict(torch.load("E:\COVID-19 DETECTION\Classification using ResNet-18\covid_classification_model2.pth"))
# model.eval()

# # Create a new model with the correct final layer
# new_model = models.resnet18(pretrained=True)
# new_model.fc = nn.Linear(new_model.fc.in_features, 2)  # Adjust to match the desired output units

# # Copy the weights and biases from the loaded model to the new model
# new_model.fc.weight.data = model.fc.weight.data[0:2]  # Copy only the first 2 output units
# new_model.fc.bias.data = model.fc.bias.data[0:2]



# Prepare your new image for classification. You should use the same data transformations you used during training. Here's an example of how to prepare an image for inference:

# In[4]:


# # Load and preprocess the unseen image
# image_path = "E:\COVID-19 DETECTION\Classification using ResNet-18\COVID-test.png"  # Replace with the path to your image
# image = Image.open(image_path)
# preprocess = transforms.Compose([
#     transforms.Resize(229*229),
# #     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# input_tensor = preprocess(image)
# input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension


# Perform inference using the model:

# In[5]:


# # Perform inference
# with torch.no_grad():
#     output = model(input_batch)

# # Get the predicted class
# _, predicted_class = output.max(1)

# # Map the predicted class to the class name
# class_names = ['COVID', 'Normal']  # Make sure these class names match your training data
# predicted_class_name = class_names[predicted_class.item()]

# print(f'The predicted class is: {predicted_class_name}')


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Display the image with the predicted class name
image = np.array(image)
plt.imshow(image)
plt.axis('off')
plt.text(10, 10, f'Predicted: {predicted_class_name}', fontsize=12, color='white', backgroundcolor='red')
plt.show()

