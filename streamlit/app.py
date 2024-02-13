import streamlit as st
import torch 
import torchvision
from torchvision import transforms
from PIL import Image

#####
###
# Initialization
###
#####
if 'generate_result' not in st.session_state:
    st.session_state['generate_result'] = 0
if 'show_result' not in st.session_state:
    st.session_state['show_result'] = 0
if 'number_of_files' not in st.session_state:
    st.session_state['number_of_files'] = 0
if 'upload_choice' not in st.session_state:
    st.session_state['upload_choice'] = 'file_up'


#####
###
# Used to show either the file_uploader or the webcam
###
#####
def change_state():
    if st.session_state['upload_choice'] == 'file_up':
        st.session_state['upload_choice'] = 'webcam'
    else:
        st.session_state['upload_choice'] = 'file_up'

# User toggle for file_uploader vs webcam
st.toggle(label="Webcam",  help="Click on to use webcam, off to upload a file", on_change=change_state)   

# Use state to know whether to show file_uploader or webcam
if st.session_state['upload_choice'] == 'file_up':
    img = st.file_uploader(label="Upload a photo of a squirrel or bird", type=['png', 'jpg'])
    if img is not None:
        st.session_state['number_of_files'] = 1
    else:
        st.session_state['number_of_files'] = 0
else:
    img = st.camera_input(label="Webcam")
    if img is not None:
        st.session_state['number_of_files'] = 1
    else:
        st.session_state['number_of_files'] = 0


#####
###
# Load the image and apply transformations
###
#####
def predict_image(image_path, model):

    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move input tensor to the device (GPU if available)
    input_image = input_image.to('cpu')

    # Perform inference
    model.eval()
    with torch.no_grad():
        output = model(input_image)

    # Get predicted class probabilities and class index
    probabilities = torch.softmax(output, dim=1)[0]
    predicted_class_index = torch.argmax(probabilities).item()

    # Map class index to class label
    class_labels = ["Bird", "Squirrel"]
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label
#     print("Class probabilities:")
#     for i, prob in enumerate(probabilities):
#         print(f"{class_labels[i]}: {prob:.4f}")


#####
###
# Load model and prepare for inference
###
#####
model_loaded = torchvision.models.resnet18(pretrained=False)  # Initialize ResNet18 without pretraining 
model_loaded.fc = torch.nn.Linear(model_loaded.fc.in_features, 2)  # Modify the fully connected layer
model_loaded = model_loaded.to('cpu')  # Move the model to the appropriate device (GPU or CPU)

# Load the saved state dictionary into the model
model_path = 'resnet18_custom_model.pth'
model_loaded.load_state_dict(torch.load(model_path, map_location='cpu'))

# Set the model to evaluation mode
model_loaded.eval()


#####
###
# Toggle view of model output in UI
###
#####
if st.session_state['upload_choice'] == 'file_up' and st.session_state['number_of_files'] == 1:
    st.session_state['generate_result'] = 1
    st.session_state['show_result'] = 1
elif st.session_state['upload_choice'] == 'webcam' and st.session_state['number_of_files'] == 1:
    st.session_state['generate_result'] = 1
    st.session_state['show_result'] = 1   
else:
    st.session_state['generate_result'] = 0
    st.session_state['show_result'] = 0



if st.session_state['generate_result'] != 0:
    if img is not None:
        result = predict_image(image_path=img, model=model_loaded)
    st.session_state['generate_result'] = 0

if st.session_state['show_result'] != 0:
    if result == 'Bird':
        st.markdown("""
            <style>
            .centered {
                text-align: center;
            }
            </style>
            <div class="centered">
                🐦🐦🐦🐦🐦🐦🐦🐦🐦🐦🐦🐦🐦🐦🐦🐦
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
            <style>
            .big-font {
                font-size:30px !important;
                text-align: center;
            }
            </style>
            <div class="big-font">
            That's a Bird
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <style>
            .centered {
                text-align: center;
            }
            </style>
            <div class="centered">
                🐦🐦🐦🐦🐦🐦🐦🐦🐦🐦🐦🐦🐦🐦🐦🐦
            </div>
            """, unsafe_allow_html=True)
        if st.session_state['upload_choice'] == 'file_up':
            st.image(img)
    else:
        st.markdown("""
            <style>
            .centered {
                text-align: center;
            }
            </style>
            <div class="centered">
                🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
            <style>
            .big-font {
                font-size:30px !important;
                text-align: center;
            }
            </style>
            <div class="big-font">
            That's a Squirrel
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <style>
            .centered {
                text-align: center;
            }
            </style>
            <div class="centered">
                🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️🐿️
            </div>
            """, unsafe_allow_html=True)
        if st.session_state['upload_choice'] == 'file_up':
            st.image(img)