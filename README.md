# AIPI540-CV-Module-Project

This is the GitHub repository that contains code for the Computer Vision Module Project for AIPI 540
Authors: Suneel Nadipalli, Jared Bailey, Jay Swayambunathan 
AIPI 540 Spring 2024 
Professor: Dr. Brinnae Bent 

The objectives of this assignment were to develop a novel computer vision based project to showcase our understanding of concepts covered in class. For this assignment, we decided to create a computer vision based bird feeder than can detect birds and squirrels visiting bird feeders, and sound an alarm to chase the squirrels away. To this end, we developed a hardware system (Raspberry Pi) based which contains the motion sensor, camera, and alarm. We developed several models in parallel that can process and make predictions on animal images and classify the images as either birds or squirrels. These models include a non-neural network approach (SVM), a pre-trained mean neural network model (AlexNet), and a transfer learning based ResNet model. We then hosted our final transfer learning based ResNet model on Heroku and submitted images from our hardware system via an API call. During testing, the model successfully identified several bird/squirrel stuffed toys and the system responded appropriately by raising the alarm for the squirrel toys. 

Our Repository is Structured as Follows: 

Folders 

  data 

    > processed 

        > data_array.npy - array of all bird and squirrel images after resizing (224x224), converting to grayscale, and flattening.
        > labels_array.npy - array of labels ("Bird", "Squirrel") that correpond to images in data_array.npy file 

    > raw/bird-and-squirrels/CV_data

        > Bird - raw bird images used for model training/testing
        > Squirrel - raw squirrel images used for model training/testing

  models 

    > resnet18_custom_model.pth
    > svm_model.pkl

  
        
      

    
