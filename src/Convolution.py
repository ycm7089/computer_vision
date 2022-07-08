#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
from matplotlib import pyplot as plt

def convolution(img):
    rospy.init_node('convloution_node', anonymous=False)
    
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # son_img = processImage('Image.jpeg')

    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    # kernel = np.array([[1,2,3],[4,5,6],[7,8,9]]) # 작동안함
    # kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]]) # 빨간색의 수직 윤곽선만 떼어내는 필터
    # kernel = np.array([[0,0,0],[0,0,0],[0,0,0]]) # 초록색의 수직 윤곽선만 떼어내는 필터

    output = convolve2D(grey_img, kernel, padding=2)
    # padding 이미지 데이터의 축소를 막기 위해, Edge pixel data를 충분히 활용하기 위해
    # 참고 사이트 : https://egg-money.tistory.com/92

    # cv2.imwrite('ss', output)
    cv2.imshow('origin',img)

    cv2.imshow('greyscale',output)
    cv2.waitKey(0)

def convolve2D(image, kernel, padding=0, strides=1):
    #stride : "1"칸씩 띄워서 계산하겠다!!!

    kernel = np.flipud(np.fliplr(kernel)) # flipud : 축(위/아래)을 따라 요소의 순서를 반대로  fliplr : 왼쪽 오른쪽 방향으로 배열을 뒤집는다
    print(padding)
    xKernShape = kernel.shape[0]  # 3  kernel 3X3 matrix
    yKernShape = kernel.shape[1] # 3
    xImgShape = image.shape[0] #733
    yImgShape = image.shape[1] #620
    
    print(yKernShape)
    print(xKernShape)
    print(xImgShape)
    print(yImgShape)

    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    if padding !=0:
        imagePadded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

if __name__ == '__main__':
    try:
        img = cv2.imread('/home/cm/catkin_ws/src/computer_vision/sonsational.jpg')

        convolution(img)
    except rospy.ROSInterruptException:
        pass