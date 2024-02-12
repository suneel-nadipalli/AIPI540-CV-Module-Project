from scripts import make_dataset, build_features, train_models
from sklearn.model_selection import train_test_split

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import torch
# Load the data

make_dataset.download_data()

# Build the features

data_array, labels_array = build_features.build_feat_svm('.\\data\\raw\\birds-and-squirrels\\CV_data')

X_train, X_test, y_train, y_test = train_test_split(data_array, labels_array, test_size=0.2, random_state=0)

svm_model = train_models.train_svm(X_train, X_test, y_train, y_test)

import pickle

# save
with open('models\svm_model.pkl','wb') as f:
	   pickle.dump(svm_model,f)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = build_features.CustomDataset_ResNet(root_dir='data\\raw\\birds-and-squirrels\CV_data', transform=transform, percent=100)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


resnet_model = train_models.train_resnet(train_loader, test_loader)

model_path = 'models\\resnet18_custom_model.pth'

# Save the model
torch.save(resnet_model.state_dict(), model_path)

