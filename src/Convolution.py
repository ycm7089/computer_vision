#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
from matplotlib import pyplot as plt

def convolution():
    img1 = cv2.imread('/home/cm/catkin_ws/src/computer_vision/sonsational.jpg')
    img6 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    cv2.imshow('greyscale',img6)
    cv2.waitKey(0)





if __name__ == '__main__':
    try:
        convolution()
    except rospy.ROSInterruptException:
        pass