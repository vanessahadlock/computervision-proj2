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
                cornerImg[y, x] = R  
                
    return cornerImg

# Performs NMS on corner imaage given a window size
def nonMaxSupression(cornerImg, wsize):

    if wsize == 0:
        return cornerImg

    if wsize % 2 == 0:
        wsize += 1
    offset = wsize // 2

    h, w = cornerImg.shape
    for y in range(offset, h - offset):
        for x in range(offset, w - offset):

            rWindow = cornerImg[y - offset : y + offset + 1, x - offset : x + offset + 1]            

            rWindow[rWindow < np.max(rWindow)] = 100  
    
    return cornerImg

# Returns binary corner image (0 or 255)
def harrisNMS(img, alpha, wSizeHarris, wSizeNMS, threshold):

    
    cornerImg = harrisCorners(img, wSizeHarris, alpha, threshold) 
    
    cornerImg = nonMaxSupression(cornerImg, wSizeNMS) 
        
    return cornerImg

def SSD(f, g):
    return np.sum((f - g) ** 2)

def CC(f, g):
    f = np.int32(f)
    g = np.int32(g)

    return np.sum(f * g)

def NCC(f, g):

    f = np.int32(f)
    g = np.int32(g)

    # print(f'f\n{f}\n')
    # print(f'g:\n{g}\n')

    # print(f'f ** 2:\n{f ** 2}\n')
    # print(f'g ** 2:\n{g ** 2}\n')
    
    fmag = np.sum(f ** 2) ** (1/2)
    gmag = np.sum(g ** 2) ** (1/2)

    # print(f'fmag:\n{fmag}\n')
    # print(f'gmag:\n{gmag}\n')

    # print(f'f / fmag:\n{f / fmag}\n')
    # print(f'g / gmag:\n{g / gmag}\n')

    # print(f'nCC: {np.sum((f / fmag) * (g / gmag))}')

    return np.sum((f / fmag) * (g / gmag))

def correspondanceNCC(img1, img2, corners_img1, corners_img2, wsize, threshold):

    img1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if wsize % 2 == 0:
        wsize += 1
    offset = wsize // 2

    h1, w1 = corners_img1.shape
    h2, w2 = corners_img2.shape

    corners1 = []
    for y1 in range(offset, h1 - offset):
        for x1 in range(offset, h1 - offset):
            if corners_img1[y1][x1] == 255:
                corners1.append([y1, x1])

    corners2 = []          
    for y2 in range(offset, h2 - offset):
        for x2 in range(offset, w2 - offset):
            if corners_img2[y2][x2] == 255:
                corners2.append([y2, x2])

    correspondances = []
    ncc = []
    count = 0
    for coord1 in corners1:

        nccList = []
        for coord2 in corners2:
            
            y1, x1 = coord1
            y2, x2 = coord2 

            w1 = img1[y1 - offset : y1 + offset + 1, x1 - offset : x1 + offset + 1]
            w2 = img2[y2 - offset : y2 + offset + 1, x2 - offset : x2 + offset + 1]

            # nccList.append(NCC(w1, w2))
            nccList.append(SSD(w1, w2))       

        # print(f'nccList:\n{np.round(nccList, 5)}\n')

        # nccMax = max(nccList)
        # print(f'nccMax:\n{nccMax}\n')
        # print(f'nccList.index(nccMax):\n{nccList.index(nccMax)}\n')

        # if nccMax > threshold:
        #     pt2 = corners2[nccList.index(nccMax)]
        #     correspondences.append(coord1 + pt2)
        #     ncc.append(nccMax)

        nccMin = min(nccList)
        # print(f'nccMin:\n{nccMin}\n')
        # print(f'nccList.index(nccMin):\n{nccList.index(nccMin)}\n')

        if nccMin > threshold:
            pt2 = corners2[nccList.index(nccMin)]
            correspondances.append(coord1 + pt2)
            ncc.append(nccMin)

        # count += 1
        # if count == 5:
        #     return correspondences


    # print(f'correspondence:\n{correspondence}\n')
    # print(f'ncc:\n{ncc}\n')

    return correspondances

def alignImages(img1, img2, correspondances, RANSAC):
    
    # Sort matches by the NCC score
    # matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove matches that aren't scored as high
    # numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    # matches = matches[:numGoodMatches]

    keypoints1: np.ndarray = [correspondances[1], correspondances[2]]
    keypoints2: np.ndarray = [correspondances[3], correspondances[4]]

    print(f'keypoints1: {keypoints1}')
    print(f'keypoints1: {keypoints2}')
 
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


    # Print estimated homography
    print("Estimated homography : \n",  H)

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
    matchimg = cv2.drawMatches(img1, keypoints1, img2, keypoints2, correspondances, None, **draw_params)

    plt.imshow(matchimg, 'gray'), plt.show()

    return matchimg

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
    wSizeHarris = 3 # size of the window for the harris corner detection
    wSizeNMS = 0 # size of the window for NMS
    hThreshold = 1500000000 # threshold for defining what is a corner, if R > threshold

    # print("Finding corners for img1...")
    # corners_img1 = harrisNMS(img1, alpha, wSizeHarris, wSizeNMS, hThreshold)
    # cv2.imwrite("test1.jpg", corners_img1)

    # print("Finding corners for img2...")
    # corners_img2 = harrisNMS(img2, alpha, wSizeHarris, wSizeNMS, hThreshold)
    # cv2.imwrite("test2.jpg", corners_img2)

    ##################################################
    ######### Find correspondences using NCC #########
    ##################################################

    wNCC = 7 # window size for ncc
    nccThreshold = 0 # threshold for defining if two windows match

    corners_img1 = cv2.imread("test1.jpg", cv2.IMREAD_GRAYSCALE)
    corners_img2 = cv2.imread("test2.jpg", cv2.IMREAD_GRAYSCALE)

    # print(corners_img1)
    # print("")
    # print(print(corners_img2))

    print("Finding correspondences...\n")
    correspondances = correspondanceNCC(img1, img2, corners_img1, corners_img2, wNCC, nccThreshold)
    print(f'correspondences:\n{correspondances}\n')   

    ##################################################
    ######## Estimate homography using RANSAC ########
    ##################################################

    ##################################################
    ############# Align images into one ##############
    ##################################################

    print("Aligning images ...")
    # Takes in the two images and the correspondance list 
    # as well as true or false for RANSAC to determine if
    # we want to remove the outliers or not
    RANSAC = True

    panorama = alignImages(img1, img2, correspondances, RANSAC)

    # Write aligned image to disk.
    outFilename = "panorama.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, panorama)


if __name__ == "__main__":
    main()