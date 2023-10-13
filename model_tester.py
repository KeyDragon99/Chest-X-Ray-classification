from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import time
import os

# Give model's file path
model_path = 'C:/Users/simos/Desktop/Covid19-dataset/saved_models/inceptionv3covid-32-50-Pretrain.pt'
# Give model's name
model_name = 'inception'
# Give model's number of output classes
num_classes = 3
# Give input size for the model
input_size = 299
# Give the directory of the dataset
data_dir = 'C:/Users/simos/Desktop/Covid19-dataset/'
# Give a batch size
batch_size = 32
# Set device to cuda if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def initialize_model(model_name, model_path):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        # model_ft = models.inception_v3(pretrained=use_pretrained)
        # # set_parameter_requires_grad(model_ft, feature_extract)
        # # Handle the auxilary net
        # num_ftrs = model_ft.AuxLogits.fc.in_features
        # model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # # Handle the primary net
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = 299
        # model_ft.load_state_dict(torch.load(model_path, torch.device('cpu')))  # Save the network model
        model_ft = torch.load(model_path)
        model_ft.eval()  # Set the network to evaluation mode

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Data normalization for validation
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create validation dataset
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                  ['val']}
# Create validation dataloader
dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in
    ['val']}

def test_model(model, dataloaders, criterion):
    since = time.time()

    # Each epoch has a training and validation phase
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloaders['val']:
        with torch.set_grad_enabled(False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            model = model.to(device)
            # Get model outputs and calculate loss
            # Special case for inception because in training it has an auxiliary output.
            # In testing we only consider the final output.
            outputs = model(inputs)
            # print(outputs)
            # print(labels)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloaders['val'].dataset)
    epoch_acc = running_corrects.double() / len(dataloaders['val'].dataset)

    print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    print()

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, model_path)
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()
# Run an evaluation round on the model we trained
test_model(model_ft, dataloaders_dict, criterion)
