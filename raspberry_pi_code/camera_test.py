###
###
###
# Test camera and capture for raspberry pi
###
###
###

import picamera

camera = picamera.PiCamera()

camera.capture('/home/pi/Desktop/image.jpg')