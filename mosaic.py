# Computer Vision, Project 2
# Vanessa Hadlock, Stav Rones

import numpy as np
from numpy.fft import fft2, ifft2
import cv2
import matplotlib.pyplot as plt
import math


def gradient(img):

    img = np.float32(img)

    I_dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
    I_dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
    
    # Determine M = [Ixx Ixy ; Ixy Iyy] performing element-wise multiplication
    I_xx = np.square(I_dx)
    I_yy = np.square(I_dy)
    I_xy = np.multiply(I_dx, I_dy)

    # # Calculate determinant
    # det = I_xx * I_yy - (I_xy ** 2)

    # # Calculate trace (alpha = 0.04 to 0.06)
    # alpha = 0.04
    # trace = alpha * (I_xx + I_yy) ** 2

    # # Using determinant and trace, calculate corner response function
    # R = det - trace

    # cornerList = cv2.harrisCorners(img, height, width, Ixx, Iyy, Ixy, k, offset, threshold)

    # Display corner response function
    # f, ax1 = plt.subplots(1, 1, figsize=(20,10))

    # ax1.set_title('Corner response fuction')
    # ax1.imshow(R, cmap="gray")
    return I_xx, I_yy, I_xy


def nonMaxSupression(cornerList, windowsize):

    #Sort in decreasing order

    cornerList.sort(reverse=True)
    
    #Mark unwanted neighbors based on Window Size
    for i in cornerList:
        index = []
        if i[3] != 1:
            for j in cornerList:
                if j[3] != 1:
                    dist = math.sqrt((j[1] - i[1])**2 + (j[2] - i[2])**2)
                    if (dist <= windowsize and dist > 0):
                        j[3] = 1
    
    #Filter out neighbors
    final = filter(lambda x: x[3] == 0, cornerList)   
    
    return list(final)


def cornerDetection(img, alpha, windowsize, threshold):
    
    plot_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height = img.shape[0]
    width = img.shape[1]
    
    # Calculate window offset
    if windowsize % 2 == 0:
        windowsize += 1
    offset = windowsize//2
    
    # Compute the gradient
    I_xx, I_yy, I_xy = gradient(img)

    cornerList = harrisCorners(height, width, I_xx, I_yy, I_xy, alpha, offset, threshold) 
    corners = nonMaxSupression(cornerList, windowsize) 
        
    return corners


# Harris Corner Detector
def harrisCorners(height, width, Ixx, Iyy, Ixy, alpha, offset, threshold):

    cornerList = []
    
    #Scan T-B, L-R
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            
            #Get windows
            Wxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            Wyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Wxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            
            #Calculate sum of squares
            Sxx = Wxx.sum()                
            Syy = Wyy.sum()    
            Sxy = Wxy.sum()                
            
            #Calculate determinant, trace, and cornerness
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            g = det - alpha*(trace**2)
            
            #If cornerness > threshold, add to corner list
            if g > threshold:
                cornerList.append([g, x, y, 0])               
                
    return cornerList

    
# def corner(img):
        
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # find Harris corners

#     # convert gray image into floating point numbers in order
#     # to put it into the Harris Corner Detector 
#     gray = np.float32(gray)

#     # run grayscale float32 image through corner detector
#     # blockSize = 2, ksize = 3, k = 0.04 
#     dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    
#     # print(dst)

#     #result is dilated for marking the corners
#     dst = cv2.dilate(dst,None)

#     # Threshold for an optimal value, it may vary depending on the image.
#     ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)

#     img[dst>0.01*dst.max()]=[0,0,255]

#     cv2.imshow('dst', img)

#     if cv2.waitKey(0) & 0xff == 27:
#         cv2.destroyAllWindows()

#     # Conver the image back into 8 bit integers
#     dst = np.uint8(dst)

#     # find centroids
#     ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

#     # define the criteria to stop and refine the corners
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
#     corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

#     # Now draw the corners onto the image 
#     res = np.hstack((centroids,corners))
#     res = np.int0(res)
#     img[res[:,1],res[:,0]]=[0,0,255]
#     img[res[:,3],res[:,2]] = [0,255,0]

#     return img

def main(): 

    # constant between 0.04 - 0.06
    alpha = 0.04
    windowSize = 5
    threshold = 2000

    # read in first image
    img1: np.ndarray = cv2.imread('DanaHallWay1/DSC_0281.jpg') 
    # cv2.imwrite('DanaHallWay1/DSC_0281_grayscale.jpg', img1)
    
    # perform Harris Corner Detection on the first image
    # corner1 = corner(img1)
    # cv2.imwrite('DanaHallWay1/DSC_0281_corner.jpg', corner1)

    corners = cornerDetection(img1, alpha, windowSize, threshold)
     
    # cv2.show(gmax1)

    # read in second image
    img2: np.ndarray = cv2.imread('DanaHallWay1/DSC_0282.jpg') 
    
    # cv2.imwrite('DanaHallWay1/DSC_0282_grayscale.jpg', img2)
    # perform Harris Corner Detection on the second image
    # corner2 = cornerDetection(img2)
    # cv2.imwrite('DanaHallWay1/DSC_0282_corner.jpg', corner2)

    corners = cornerDetection(img2, alpha, windowSize, threshold)    

    # read in third image as a grayscale
    # img3: np.ndarray = cv2.imread('DanaHallWay1/DSC_0283.jpg', cv2.IMREAD_GRAYSCALE) 
    # cv2.imwrite('DanaHallWay1/DSC_0283_grayscale.jpg', img3)
    # cornerDetection(img3)
    # cv2.imwrite('DanaHallWay1/DSC_0282_grayscale.jpg', img2)



if __name__ == "__main__":
    print("start of program")
    main()