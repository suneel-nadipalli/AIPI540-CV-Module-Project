###
###
###
# This is a python script to take sample night time photos using a night enabled camera
# The camera is controlled by a Raspberry Pi
# These photos will be used to:
#   - help with automated conversion of the camera images to the model desired format
#   - help converted training images to a similar format to the night time photos
###
###
###

from picamera import PiCamera
from time import sleep

camera = PiCamera()
path = '/home/pi/Desktop/'
i = 0

while True:
    i += 1
    camera.capture(path + 'pic' + str(i) + '.jpg')
    sleep(1)

    if i == 20:
        break