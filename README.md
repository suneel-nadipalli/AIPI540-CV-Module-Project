# AIPI540-CV-Module-Project

This is the GitHub repository that contains code for the Computer Vision Module Project for AIPI 540

Authors: Suneel Nadipalli, Jared Bailey, Jay Swayambunathan 

AIPI 540 Spring 2024 

Professor: Dr. Brinnae Bent 

The objectives of this assignment were to develop a novel computer vision based project to showcase our understanding of concepts covered in class. For this assignment, we decided to create a computer vision based bird feeder than can detect birds and squirrels visiting bird feeders, and sound an alarm to chase the squirrels away. To this end, we developed a hardware system (Raspberry Pi) based which contains the motion sensor, camera, and alarm. We developed several models in parallel that can process and make predictions on animal images and classify the images as either birds or squirrels. These models include a non-neural network approach (SVM), a pre-trained mean neural network model (AlexNet), and a transfer learning based ResNet model. We then hosted our final transfer learning based ResNet model on Heroku and submitted images from our hardware system via an API call. During testing, the model successfully identified several bird/squirrel stuffed toys and the system responded appropriately by raising the alarm for the squirrel toys. 

Running the tool:

Users will need a Raspberry Pi 4 running a 32 bit Debian OS. Users will connect the GPIO pins to the appropriate wires and sensors as displayed numerically in the main.py file. The main.py file is the only file needed to run the tool. 

Hardware needs include: motion sensor, camera (in Pi camera slot), and power relay. The power relay is used to activate an ultrasonic speaker of the user's choice.

Users will need to access the model file and output through an API call, as the Raspberry Pi cannot host this model running a 32 bit OS.

Streamlit App: 

https://huggingface.co/spaces/JaredBailey/BirdOrSquirrelV1

Several sample images tested in the app were captured using the rasperry pi day and nighttime camera. These images can be found in the data folder.

Our Repository is Structured as Follows: 

Folders 

  > data 

    > processed 

        > data_array.npy - array of all bird and squirrel images after resizing (224x224), converting to grayscale, and flattening.
        > labels_array.npy - array of labels ("Bird", "Squirrel") that correpond to images in data_array.npy file 

    > raw/bird-and-squirrels/CV_data

        > Bird - raw bird images used for model training/testing
        > Squirrel - raw squirrel images used for model training/testing
  > models 

    > resnet18_custom_model.pth - original transfer learning approach with ResNet (NOT FINAL MODEL) 
    > svm_model.pkl - non-neural network approach SVM model 

  > notebooks

    > Data_Preparation.ipynb - notebook delineating steps to prepare data including resizing with skimage (224,224), converting images to greyscale to work with nightvision/IR images, flattening images, and storing images and labels as numpy arrays data_array.npy and 
    labels_array.npy 

    > Data_Splitting_&_SVM.ipynb - notebook describing steps to split the data (80% training, 20% testing using sklearn train_test_split), run data through non-neural network model (SVM), and output predictions/accuracy metrics for test set analysis

    > DL_FineTuning_Model.ipynb - notebook containing code for downloading pre-trained ResNet model, augmenting training with our own bird and squirrel training images, and testing accuracy of transfer learning approach on test set. 

    > Mean_Model.ipynb - notebook showing the steps involved in downloading the mean model comparitor (AlexNet) and running test images through this model to calculate classification accuracy. Also contains code to prepare, split, and train/test SVM and transfer learning approach to facilitate comparisons between these methods. 

  > raspberry_pi_code

      > build_code

        > night_time_sample_image_gather.py - code for generating nighttime sample images to test system's performance in low light conditions 

      > test_code 

        > motion_sensor_test.py - code for testing motion sensor's ability to detect moving objects in field of view 

        > camera_test.py - code for testing camera's ability to capture images of sufficient quality and size 

  > scripts

      > build_features.py - script that builds features that will be run through the SVM, mean model, and transfer learning approach 

      > make_datasets.py - script that downloads aggregate/raw bird and squirrel dataset from the storage location on Kaggle

      > train_models.py - script that trains the SVM and ResNet models, and returns trained models along with predictions on test set 


  > setup.py - script that calls each step of the data preparation and analysis pipeline sequentially


  > streamlit

      > app.py - script containing Raspberry Pi controls as well as UI interface (streamlit) 

    
  Final Results: 

  After testing the SVM model (non-neural network approach), the AlexNet model (pre-trained mean model), and the transfer learnign based ResNet approach, we determined that the transfer learning approach provided the most accurate and robust model for our purpose. This model was hosted online using the platform Heroku, and an API call was structured to submit motion-capture images from our hardware apparatus, 
  and generate predictions. These predictions would be sent back to our hardware apparatus (Raspberry Pi based system) and the appropriate action would be taken (none for birds vs. alarm to antagonize squirrels)

  Future Directions: 

  To continue testing the model in a variety of conditions it may encounter in the real world (inclement weather, other animals making an appearance, motion artifacts, etc.) to ensure that its performance is robust. Potentially developing an application that alerts customers whenever an animal is detected at their bird feeder by pushing a notification to their phones.

