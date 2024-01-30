###
###
###
# This is a python script to make use of a sensor to detect motion
###
###
###

from gpiozero import MotionSensor
from time import sleep

pir = MotionSensor(16)

while True:
	print("Scanning for motion...")
    pir.wait_for_motion()
    print("Motion Detected")
	sleep(1)
	pir.wait_for_no_motion()
	print("Motion Stopped")