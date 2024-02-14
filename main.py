###
# This script runs on a Raspberry Pi 4 with a night vision camera and a motion sensor
# The camera takes a photo when motion is detected and sends it to a server for classification
# If the classification is not a bird, the relay is activated to scare the animal
# The relay is connected to an ultrasonic frequency alarm
###

#####
###
# 00 Imports
###
#####
from gpiozero import MotionSensor
from picamera import PiCamera
import RPi.GPIO as GPIO
from time import sleep
import requests




####
###
# 01 Variable Setup
###
#####
pir = MotionSensor(15) # GPIO Pin
camera = PiCamera()
i = 1 # photo counter for naming
path = "/home/jaredbailey/Desktop/Home/Projects/Bird_Feeder/Images/" # raspberry pi path
pin = 14 # power relay pin
# configure pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)




####
###
# 02 Functions
###
#####
def take_photo(path):
    """
    Takes a photo using the night vision camera
    
    Args
        Path to save photo - string
    Return
        Name of the image - string
    """
    global i
    camera.capture(path + "pic_" + str(i) + ".jpg")
    print(f"Photo {i} taken")
    image_name = "pic_" + str(i) + ".jpg"
    i += 1
    sleep(1) # one second rest between photos
    return image_name


def send_receive_photo(path, image_name):
    """
    Sends photo to server for classification
    Receives classification back if photo was bird(1) or another animal(0)
    
    Args
        path - string
        photo - jpg
    Return
        classification - boolean
    """
    classification = 0
    
    print(path + image_name)
    
    try:
        response = requests.post(
        "https://cv-proj-api-bf845c15625a.herokuapp.com/predict",
        files={
            "image": open(path + image_name, "rb")
            }
        )
        
        output = response.json()
        print(output)
        classi = output["pred_class"]
        
        if classi == "bird":
            classification = 0
        else:
            classification = 1
    
    except:
        pass
    
    return classification


def scare_animal(pin, classification=1):
    """
    Scare animal with ultrasonic frequency alarm
    
    Args
        classification - 1 for bird, 0 for other animal - boolean
    Return
        None
    """
    
    if classification == 1:
        try:
            # activate relay
            GPIO.output(pin, GPIO.HIGH) 
            print("Relay Activated")
            
            # power on for 10 seconds
            sleep(10)
            
            # deactivate relay
            GPIO.output(pin, GPIO.LOW)
            print("Relay Deactivated")
            
        finally:
            # GPIO.cleanup() # return pin to default state
            pass
    
    
def all_together_pipeline():
    """
    Combine previous functions into pipeline
    
    Args
        None
    Return
        None
    """
    image_name = take_photo(path=path)
    classification = send_receive_photo(path=path, image_name=image_name)
    scare_animal(pin=pin, classification=classification)
    




####
###
# 03 Action
###
#####
while True:
    pir.when_motion = all_together_pipeline
