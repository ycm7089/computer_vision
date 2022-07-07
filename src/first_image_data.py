#!/usr/bin/env python3

import cv2
import rospy


def read():
    img1 = cv2.imread('/home/cm/catkin_ws/src/computer_vision/sonsational.jpg')
    # img1 = cv2.resize(img1,(640, 480))
    img2 = cv2.imread('/home/cm/catkin_ws/src/computer_vision/sonsational.jpg',cv2.IMREAD_GRAYSCALE)
    # img3 = cv2.imread('/home/cm/catkin_ws/src/computer_vision/sonsational.jpg',cv2.IMREAD_REDUCED_COLOR_4)
    # img4 = cv2.imread('/home/cm/catkin_ws/src/computer_vision/sonsational.jpg',cv2.IMREAD_LOAD_GDAL)
    img5 = img1 + 50 #이미지 밝게
    img6 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    
    w = img1.shape[1] # 이미지 가로 사이즈
    h = img1.shape[0] # 이미지 세로 사이즈
    print(w)
    print(h)

    width = round(w/2)
    height = round(h/2)
    channel = img1.shape
    channel2= img6.shape
    # img1 = cv2.resize(img1,(640,720))
    
    # w = img1.shape[1] # 이미지 가로 사이즈
    # h = img1.shape[0] # 이미지 세로 사이즈
    # print(w)
    # print(h)
    
    img = cv2.resize(img1,(width,height))
    esized = cv2.resize(img1,(width,height), interpolation = cv2.INTER_AREA)

    cv2.imshow('image1', img1)
    cv2.imshow('image2', img)
    cv2.imshow('image', esized)
    cv2.imshow('image5', img5)
    cv2.imshow('image6', img6)
    print(channel)
    print(channel2)

    # cv2.imshow('image2', img2)
    # cv2.imshow('image3', img3)
    # cv2.imshow('image4', img4)
    cv2.waitKey(0)


if __name__ == '__main__':
    try:
        read()
    except rospy.ROSInterruptException:
        pass