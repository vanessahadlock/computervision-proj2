# Computer Vision, Project 2
# Vanessa Hadlock, Stav Rones

import numpy as np
from numpy.fft import fft2, ifft2
import cv2
import matplotlib.pyplot as plt
import math

# Finds the corner of an image using Harris algorithm
def harrisCorners(img, wsize, alpha, threshold):

    ### Compute I_xx, I_yy, I_xy ###
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.float32(img)

    if wsize % 2 == 0:
        wsize += 1
    offset = wsize // 2

    I_dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize = wsize)
    I_dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize = wsize)

    I_xx = np.square(I_dx)
    I_yy = np.square(I_dy)
    I_xy = np.multiply(I_dx, I_dy)

    ### Find cornders using threshold of R ###

    cornerImg = np.zeros_like(img)

    h, w, = img.shape
    for y in range(offset, h - offset):
        for x in range(offset, w - offset):

            # Get windows
            Wxx = I_xx[y - offset : y + offset + 1, x - offset : x + offset + 1]
            Wyy = I_yy[y - offset : y + offset + 1, x - offset : x + offset + 1]
            Wxy = I_xy[y - offset : y + offset + 1, x - offset : x + offset + 1]

            # Calculate sum of squares
            Sxx = Wxx.sum()                
            Syy = Wyy.sum()    
            Sxy = Wxy.sum()
            
            # Calculate determinant, trace, and corner response
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            R = det - alpha * (trace ** 2)
            
            # If corner response > threshold, add to corner list
            if R > threshold:
                cornerImg[y][x] = R  
                
    return cornerImg

# Performs NMS on corner imaage given a window size
def nonMaxSupression(cornerImg, wsize):

    wsize -= 2
    if wsize % 2 == 0:
        wsize += 1
    offset = wsize // 2

    h, w, = cornerImg.shape
    for y in range(offset, h - offset):
        for x in range(offset, w - offset):

            rWindow = cornerImg[y - offset : y + offset + 1, x - offset : x + offset + 1]
            rMax = np.max(rWindow)

            for r in range(wsize):
                for c in range(wsize):

                    if rWindow[r][c] < rMax:
                        cornerImg[y - offset + r, x - offset + c] = 0
    
    return cornerImg

# Returns binary corner image (0 or 255)
def harrisNMS(img, alpha, wSizeHarris, wSizeNMS, threshold):
    
    cornerImg = harrisCorners(img, wSizeHarris, alpha, threshold) 
    
    cornerImg = nonMaxSupression(cornerImg, wSizeNMS) 
        
    return cornerImg

def main(): 

    ##################################################
    ############### Read in two images ###############
    ##################################################

    img1: np.ndarray = cv2.imread('DanaHallWay1/DSC_0281.jpg') 
    img2: np.ndarray = cv2.imread('DanaHallWay1/DSC_0282.jpg')

    ##################################################
    ### Detect corner pixels using Harris with NMS ###
    ##################################################

    alpha = 0.04 # constant between 0.04 - 0.06
    wSizeHarris = 5 # size of the window for the harris corner detection
    wSizeNMS = 5 # size of the window for NMS
    threshold = 1500000000 # threshold for defining what is a corner, if R > threshold

    print("Finding corners for img1...")
    corners_img1 = harrisNMS(img1, alpha, wSizeHarris, wSizeNMS, threshold)
    cv2.imwrite("test1.jpg", corners_img1)

    print("Finding corners for img2...")
    corners_img2 = harrisNMS(img2, alpha, wSizeHarris, wSizeNMS, threshold)
    cv2.imwrite("test2.jpg", corners_img2)

    ##################################################
    ######### Find correspondences using NCC #########
    ##################################################

    ##################################################
    ######## Estimate homography using RANSAC ########
    ##################################################

    ##################################################
    ############# Align images into one ##############
    ##################################################

if __name__ == "__main__":
    main()