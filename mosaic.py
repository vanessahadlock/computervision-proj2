# Computer Vision, Project 2
# Vanessa Hadlock, Stav Rones

import numpy as np
from numpy.fft import fft2, ifft2
import cv2
import matplotlib.pyplot as plt
import math

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

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
            
            #Calculate determinant, trace, and corner response
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            g = det - alpha*(trace**2)
            
            #If corner response > threshold, add to corner list
            if g > threshold:
                cornerList.append([g, x, y, 0])               
                
    return cornerList


def RANSAC():


    return

    
def normalizedCrossCorrelation(img1, img2):

    return


## this found th corners using the cornerHarris function, but it doesn't seem to return what we need it to 
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

# Calculates the homography using RANSAC and then warps the image together
# Takes in the 4 points of correspondances between the two images

def alignImages(img1, img2, matches, RANSAC):
    
    # Sort matches by the NCC score
    # matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove matches that aren't scored as high
    # numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    # matches = matches[:numGoodMatches]

    keypoints1: np.ndarray = [matches[1], matches[2]]
    keypoints2: np.ndarray = [matches[3], matches[4]]
 
    # # Draw top matches
    # imMatches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, correspondances, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # # Extract location of good matches
    # points1 = np.zeros((len(correspondances), 2), dtype=np.float32)
    # points2 = np.zeros((len(correspondances), 2), dtype=np.float32)

    # for i, match in enumerate(correspondances):
    #     points1[i, :] = keypoints1[match.queryIdx].pt
    #     points2[i, :] = keypoints2[match.trainIdx].pt

    # # Use homography
    # height, width, channels = img2.shape
    # im1Reg = cv2.warpPerspective(img1, h, (width, height))

    # get the 3x3 transformation homography 

    if bool(RANSAC):
        H, mask = cv2.findHomography(keypoints1, keypoints2, cv2.RANSAC)

    else:
        H, mask = cv2.findHomography(keypoints1, keypoints2)
    
    # inliers of the RANSAC
    matchesMask = mask.ravel().tolist()

    # get the size of image 1 
    height, width = img1.shape

    # creating a copy of img1 using its dimensions  
    imgcopy = np.float32([ [0,0],[0,height-1],[width-1,height-1],[width-1,0] ]).reshape(-1,1,2)
    
    # run perspective transform to warp the img copy on the output
    # using the homograpy 
    dst = cv2.perspectiveTransform(imgcopy, H)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    # draw the inliners of the RANSAC  
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    # draw the matches between the two images 
    matchimg = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, **draw_params)

    plt.imshow(matchimg, 'gray'), plt.show()

    return matchimg


def main(): 

    # constant between 0.04 - 0.06
    alpha = 0.04

    # size of the window for the harris corner detection
    windowSize = 5

    # threshold for defining what is a corner, if R > threshold
    threshold = 2000

    # Set RANSAC equal to true or false depending on if we want to clean up the
    # outlying matches or not 
    RANSAC = 'True'

    # read in first image
    img1: np.ndarray = cv2.imread('DanaHallWay1/DSC_0281.jpg') 
    # cv2.imwrite('DanaHallWay1/DSC_0281_grayscale.jpg', img1)
    
    # perform Harris Corner Detection on the first image
    # this returns the image with the red dots indicating where the corners are
    # we can use this to test the threshold value
    # corner1 = corner(img1)
    # cv2.imwrite('DanaHallWay1/DSC_0281_corner.jpg', corner1)

    corners_img1 = cornerDetection(img1, alpha, windowSize, threshold)
    
    # print(corners)

    # read in second image
    img2: np.ndarray = cv2.imread('DanaHallWay1/DSC_0282.jpg') 
    
    # cv2.imwrite('DanaHallWay1/DSC_0282_grayscale.jpg', img2)
    # perform Harris Corner Detection on the second image
    # this returns the image with the red dots indicating where the corners are
    # we can use this to test the threshold value
    # corner2 = cornerDetection(img2)
    # cv2.imwrite('DanaHallWay1/DSC_0282_corner.jpg', corner2)

    corners_img2 = cornerDetection(img2, alpha, windowSize, threshold)    

    ######## Just focusing on the first two imagees for now ##########
    # read in third image as a grayscale
    # img3: np.ndarray = cv2.imread('DanaHallWay1/DSC_0283.jpg', cv2.IMREAD_GRAYSCALE) 
    # cv2.imwrite('DanaHallWay1/DSC_0283_grayscale.jpg', img3)
    # cornerDetection(img3)
    # cv2.imwrite('DanaHallWay1/DSC_0282_grayscale.jpg', img2)


    print("Aligning images ...")
    # Takes in the two images and the correspondance list 
    # as well as true or false for RANSAC to determine if
    # we want to remove the outliers or not
    panorama = alignImages(img1, img2, correspondances, RANSAC)

    # Write aligned image to disk.
    outFilename = "panorama.jpg"
    print("Saving aligned image : ", outFilename);
    cv2.imwrite(outFilename, panorama)

    # blend pixels in areas of the overlapped part of the image

    # Print estimated homography
    print("Estimated homography : \n",  h)


if __name__ == "__main__":
    print("start of program")
    main()