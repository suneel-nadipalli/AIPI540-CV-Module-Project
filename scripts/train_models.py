# Importing required libraries and modules

from skimage.transform import resize
from sklearn import svm

import torch
import torchvision
from torchvision import transforms
from PIL import Image
import time


def train_svm (X_train, X_test, y_train, y_test):

    '''
    This function takes in the data and labels arrays and splits them into training and testing sets using sklearn's train_test_split
    20 percent of the data is used for testing and 80 percent for training

    The function then trains a support vector machine model using the training data and tests the model using the testing data

    The function then prints the accuracy, confusion matrix and classification report of the model

    The function returns the trained model

    Args:  
    
    data_array: numpy array of the data
    labels_array: numpy array of the labels

    Returns:

    svm_model: trained support vector machine model
    '''

    # Splitting the data into training and testing sets using sklearn's train_test_split

    # Training the support vector machine model with linear kernel, C = 1, and gamma = 1
    svm_model = svm.SVC(kernel='linear', C=1, gamma=1)

    # Fitting the model to the training data
    svm_model.fit(X_train, y_train)

    return svm_model

def predict_svm(svm_model, image):
    # resize the image to 224x224 pixels and preserve RGB color channels
    image = resize(image, (224, 224, 3))

    # flatten the image 
    image = image.flatten()

    data = []

    # append the image to the data list
    data.append(image)

    label = svm_model.predict(data)

    return label

def train_resnet(train_loader, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = CustomCNN(num_classes=len(dataset.classes)).to(device)

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        start_time = time.time()  # Record start time of epoch
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total * 100
        end_time = time.time()  # Record end time of epoch
        epoch_time = end_time - start_time
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time % 60)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Time: {minutes} minutes {seconds} seconds")

        return model
    
def predict_resnet(image_path, model_pth):

    model_loaded = torchvision.models.resnet18(pretrained=False)  # Initialize ResNet18 without pretraining
    model_loaded.fc = torch.nn.Linear(model_loaded.fc.in_features, 2)  # Modify the fully connected layer
    model_loaded = model_loaded.to(device)  # Move the model to the appropriate device (GPU or CPU)

    # Load the saved state dictionary into the model
    model_loaded.load_state_dict(torch.load(model_pth, map_location='cpu'))

    # Set the model to evaluation mode
    model_loaded.eval()

    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    input_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move input tensor to the device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_image = input_image.to(device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        output = model(input_image)

    # Get predicted class probabilities and class index
    probabilities = torch.softmax(output, dim=1)[0]
    predicted_class_index = torch.argmax(probabilities).item()

    # Map class index to class label
    class_labels = ['Bird', 'Squirrel']
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label, probabilities[predicted_class_index].item()