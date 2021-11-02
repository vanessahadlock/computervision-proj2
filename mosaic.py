# Computer Vision, Project 2
# Vanessa Hadlock, Stav Rones


import numpy as np
import cv2
import matplotlib.pyplot as plt


def cornerDetection(img):
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    img[res[:,1],res[:,0]]=[0,0,255]
    img[res[:,3],res[:,2]] = [0,255,0]

    return img




def main(): 

    # read in first image
    img1: np.ndarray = cv2.imread('DanaHallWay1/DSC_0281.jpg') 
    # cv2.imwrite('DanaHallWay1/DSC_0281_grayscale.jpg', img1)
    # perform Harris Corner Detection on the first image
    corner1 = cornerDetection(img1)
    cv2.imwrite('DanaHallWay1/DSC_0281_corner.jpg', corner1)



    # read in second image
    img2: np.ndarray = cv2.imread('DanaHallWay1/DSC_0282.jpg') 
    # cv2.imwrite('DanaHallWay1/DSC_0282_grayscale.jpg', img2)
    # perform Harris Corner Detection on the second image
    corner2 = cornerDetection(img2)
    cv2.imwrite('DanaHallWay1/DSC_0282_corner.jpg', corner2)


    # read in third image as a grayscale
    # img3: np.ndarray = cv2.imread('DanaHallWay1/DSC_0283.jpg', cv2.IMREAD_GRAYSCALE) 
    # cv2.imwrite('DanaHallWay1/DSC_0283_grayscale.jpg', img3)
    # cornerDetection(img3)
    # cv2.imwrite('DanaHallWay1/DSC_0282_grayscale.jpg', img2)




if __name__ == "__main__":
    print("start of program")
    main()