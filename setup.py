from scripts import make_dataset, build_features, train_models
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

import torch

import pickle

import matplotlib.pyplot as plt
import seaborn as sns

## SVM Model

# Load the data
make_dataset.download_data()

#Build the features

data_array, labels_array = build_features.build_feat_svm('.\\data\\raw\\birds-and-squirrels\\CV_data')

X_train, X_test, y_train, y_test = train_test_split(data_array, labels_array, test_size=0.2, random_state=0)

svm_model = train_models.train_svm(X_train, X_test, y_train, y_test)

# save
with open('models\svm_model.pkl','wb') as f:
	   pickle.dump(svm_model,f)

# create confusion matrix

accuracy, confusion_matrix, classification_report = train_models.evaluate_svm(svm_model, X_test, y_test)

print(f"Accuracy for the SVM model: {accuracy}")

print(f"Classification report for the SVM model:\n{classification_report}")

print("\n\n")

# plot confusion matrix

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font scale for better readability
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, square=True,
            annot_kws={'size': 12, 'weight': 'bold'})
plt.xlabel('Predicted labels', fontsize=14)
plt.ylabel('True labels', fontsize=14)
plt.title('Confusion Matrix SVM', fontsize=16)
plt.savefig('.\\data\\output\\confusion_matrix_svm.png', dpi=300, bbox_inches='tight')

## ResNet Model

# Load the data

dataset = build_features.CustomDataset_ResNet(root_dir='data\\raw\\birds-and-squirrels\CV_data', percent=100)

# Split the data into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Train the model
resnet_model = train_models.train_resnet(train_loader, test_loader)

model_path = 'models\\resnet18_custom_model.pth'

# Save the model
torch.save(resnet_model.state_dict(), model_path) 

# Evaluate the model

accuracy, confusion_matrix, classification_report = train_models.evaluate_resnet(resnet_model, test_loader)

# plot confusion matrix

# Create a heatmap

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font scale for better readability
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, square=True,
            annot_kws={'size': 12, 'weight': 'bold'})
plt.xlabel('Predicted labels', fontsize=14)
plt.ylabel('True labels', fontsize=14)
plt.title('Confusion Matrix ResNet', fontsize=16)
plt.savefig('.\\data\\output\\confusion_matrix_resnet.png', dpi=300, bbox_inches='tight')

accuracy, confusion_matrix, classification_report = train_models.eval_alexnet(test_loader=test_loader)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font scale for better readability
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, square=True,
            annot_kws={'size': 12, 'weight': 'bold'})
plt.xlabel('Predicted labels', fontsize=14)
plt.ylabel('True labels', fontsize=14)
plt.title('Confusion Matrix AlexNet', fontsize=16)
plt.savefig('.\\data\\output\\confusion_matrix_alexnet.png', dpi=300, bbox_inches='tight')

print(f"Accuracy for the AlexNet model: {accuracy}")

print(f"Classification report for the AlexNet model:\n{classification_report}")