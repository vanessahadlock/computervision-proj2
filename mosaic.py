# Computer Vision, Project 2
# Vanessa Hadlock, Stav Rones


import numpy as np
import cv2
import matplotlib.pyplot as plt



def main(): 

    # read in first image as a grayscale
    img1: np.ndarray = cv2.imread('DanaHallWay1/DSC_0281.jpg', cv2.IMREAD_GRAYSCALE) 
    # cv2.imwrite('DanaHallWay1/DSC_0281_grayscale.jpg', img1)

    # read in second image as a grayscale
    img2: np.ndarray = cv2.imread('DanaHallWay1/DSC_0282.jpg', cv2.IMREAD_GRAYSCALE) 
    # cv2.imwrite('DanaHallWay1/DSC_0282_grayscale.jpg', img2)

    # read in third image as a grayscale
    # img3: np.ndarray = cv2.imread('DanaHallWay1/DSC_0283.jpg', cv2.IMREAD_GRAYSCALE) 
    # cv2.imwrite('DanaHallWay1/DSC_0283_grayscale.jpg', img3)


if __name__ == "__main__":
    print("start of program")
    main()