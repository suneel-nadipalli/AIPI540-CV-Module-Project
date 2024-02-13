# Importing required libraries and modules

from skimage.transform import resize
from sklearn import svm

import torch
import torchvision
from torchvision import transforms
from PIL import Image
import time

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


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

def evaluate_svm(svm_model, X_test, y_test):
    # predict the labels of the test set
    y_pred = svm_model.predict(X_test)

    # calculate the accuracy of the model
    accuracy = accuracy_score(y_pred, y_test)

    conf_mat = confusion_matrix(y_test, y_pred, labels = ['Bird', 'Squirrel'])

    clf_rep = classification_report(y_test, y_pred)

    return accuracy, conf_mat, clf_rep

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
    model_loaded.eval()
    with torch.no_grad():
        output = model_loaded(input_image)

    # Get predicted class probabilities and class index
    probabilities = torch.softmax(output, dim=1)[0]
    predicted_class_index = torch.argmax(probabilities).item()

    # Map class index to class label
    class_labels = ['Bird', 'Squirrel']
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label, probabilities[predicted_class_index].item()

def evaluate_resnet(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()
    correct = 0
    total = 0

    preds_list = []
    true_list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs, 1)

            preds_list.extend(predicted.cpu().numpy())  # Append predicted labels to the list
            true_list.extend(labels.cpu().numpy()) 
            
            total += labels.size(0)
            
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total * 100

    true_list = ["Bird" if x == 0 else "Squirrel" for x in true_list]

    preds_list = ["Bird" if x == 0 else "Squirrel" for x in preds_list]

    conf_mat = confusion_matrix(true_list, preds_list, labels = ['Bird', 'Squirrel'])

    clf_rep = classification_report(true_list, preds_list)

    return test_accuracy, conf_mat, clf_rep

def eval_alexnet(test_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    import re

    '''

    This function calls a pretrained AlexNet model
    It runs the test images we created above through the network and generates predictions
    It then prints these classification predictions along with the probability percentage

    Args:

    X_test: numpy array of the test images

    Returns: None

    '''

    # initializing pretrained alexnet model
    model = torchvision.models.alexnet(pretrained=True)
    # putting model in evaluation mode
    model.eval()

    # defining transforms that will be applied to the image

    preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
    )])

    correct = 0
    total = 0

    # initializing empty list for storing predictions
    preds_list = []
    true_list = []

    # calling classes used in alexnet model for labelling
    with open('.\\scripts\\Doc4.txt') as f:
      classes = [line.strip() for line in f.readlines()]

    with torch.no_grad():
      for inputs, labels in test_loader:
          
          inputs, labels = inputs.to(device), labels.to(device)
          
          outputs = model(inputs)
          
          # _, predicted = torch.max(outputs, 1)

          _ , indices = torch.sort(outputs, descending = True)

          for idx in indices:
             preds_list.append(classes[idx[0]])

          true_list.extend(labels.cpu().numpy()) 
          
          total += labels.size(0)
    
    pattern = r'\d+:'

    preds = [re.sub(pattern, '', text) for text in preds_list]

    # Asked LLM to isolate the bird species from the AlexNet classes textfile

    bird_species = [
        'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house finch',
        'junco', 'indigo bunting', 'robin', 'bulbul', 'jay', 'magpie',
        'chickadee', 'water ouzel', 'kite', 'bald eagle', 'vulture',
        'great grey owl', 'black grouse', 'ptarmigan', 'ruffed grouse',
        'prairie chicken', 'peacock', 'quail', 'partridge', 'African grey',
        'macaw', 'sulphur-crested cockatoo', 'lorikeet', 'coucal',
        'bee eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan',
        'drake', 'red-breasted merganser', 'goose', 'black swan',
        'spoonbill', 'flamingo', 'little blue heron', 'American egret',
        'bittern', 'crane', 'limpkin', 'European gallinule', 'American coot',
        'bustard', 'ruddy turnstone', 'red-backed sandpiper', 'redshank',
        'dowitcher', 'oystercatcher', 'pelican', 'king penguin', 'albatross'
    ]

    # Asked LLM to isolate the squirrel species from the AlexNet classes textfile

    squirrel_species = [
        'squirrel', 'fox squirrel', 'marmot'
    ]

    # going through and reclassifying predictions as either bird, squirrel, or other
    pred_classes = []

    for pred in preds:
      for bird in bird_species:
        if bird in pred:
          pred_classes.append('Bird')
          break
      else:
        for squirrel in squirrel_species:
          if squirrel in pred:
            pred_classes.append('Squirrel')
            break
        else:
          pred_classes.append('Other')

    #returning purely text predictions
          
    true_list = ["Bird" if x == 0 else "Squirrel" for x in true_list]

    acc = accuracy_score(pred_classes, true_list)

    conf_mat = confusion_matrix(true_list, pred_classes, labels = ['Bird', 'Squirrel'])

    clf_rep = classification_report(true_list, pred_classes)

    return acc, conf_mat, clf_rep