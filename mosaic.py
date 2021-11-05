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

    if wsize % 2 == 0:
        wsize += 1
    offset = wsize // 2

    I_dx = cv2.Sobel(img, -1, 1, 0, ksize = wsize)
    I_dy = cv2.Sobel(img, -1, 0, 1, ksize = wsize)

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

    # wsize = 0

    if wsize == 0:
        return cornerImg

    if wsize % 2 == 0:
        wsize += 1
    offset = wsize // 2

    # cornerImg = cornerImg / 10000
    # cornerImg = np.int64(cornerImg)

    h, w = cornerImg.shape
    for y in range(offset,  h - offset):
        for x in range(offset, w - offset):

            rWindow = cornerImg[y - offset : y + offset + 1, x - offset : x + offset + 1]

            rMax = np.max(rWindow)

            if rMax != 0:

                # print(f'rWindow1:\n{rWindow}\n')

                # for r in range(wsize):
                #     for c in range(wsize):                               
                #         if (rWindow[r, c] != 0) & (rWindow[r, c] < rMax):
                #             cornerImg[y-offset+r, x-offset+c] = 0  
                #         else:
                #             cornerImg[y-offset+r, x-offset+c] = 255

                rWindow[rWindow < rMax] = 0

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

    ### convert format of corresponding points ###
    
    dst_pts = []
    src_pts = []

    for list in correspondances:
            src_pts.append([list[0], list[1]])
            dst_pts.append([list[2], list[3]])

    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)   

    # print(f'src_pts: {src_pts}')
    # print(f'dst_pts: {dst_pts}')

    # pick four random correlated points, two from the source image and two from destination
    src_keypoints = [src_pts[20], src_pts[40]]
    dst_keypoints = [dst_pts[20], src_pts[40]]

    # # Draw top matches
    # imMatches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, correspondances, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # # Extract location of good matches
    # points1 = np.zeros((len(correspondances), 2), dtype=np.float32)
    # points2 = np.zeros((len(correspondances), 2), dtype=np.float32)

    ###  get the 3x3 transformation homography ###

    if bool(RANSAC):
        H, mask = cv2.findHomography(src_keypoints, dst_keypoints, cv2.RANSAC)

    else:
        H, mask = cv2.findHomography(src_pts, dst_pts)

    print("Estimated homography : \n",  np.round(H, 3))

    # Print estimated homography
    print("Estimated homography : \n",  H)

    src_good_points = []
    dst_good_points = []

    # arbitrary distance threshold that the points need to be within from the homography
    # to be considered good points
    dist_threshold = 0.6 

    # need to determine what points are good my calculating their distance from the
    # homography line using the distance formula
    for points in dst_pts and src_pts:
        
        # ?? don't think this is right
        distance = np.norm(np.cross(H, H - src_pts[points]))/(np.norm(H))

        if distance < dist_threshold:
            src_good_points[points] = src_pts[points]
            dst_good_points[points] = dst_pts[points]


    # inliers of the RANSAC
    matchesMask = mask.ravel().tolist()

    print(f'matchesMask:\n{matchesMask}\n')


    # get the size of image 1 
    height, width = img1.shape[:-1]

    # creating a copy of img1 using its dimensions  
    imgcopy = np.float32([ [0,0],[0,height-1],[width-1,height-1],[width-1,0] ]).reshape(-1,1,2)
    
    # run perspective transform to warp the img copy on the output
    # using the homograpy 
    dst = cv2.perspectiveTransform(imgcopy, H)

    print(f'dst:\n{dst}\n')

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    cv2.imwrite("test3.jpg", img2)

    # draw the inliners of the RANSAC  
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    keypoints1 = src_pts
    keypoints2 = dst_pts
    # draw the matches between the two images 
    matchimg = cv2.drawMatches(img1, keypoints1, img2, keypoints2, correspondances, outImg = imgcopy, matchesThickness = 0.7, singlePointColor = 'green',  matchColor = 'blue')

    plt.imshow(matchimg, 'gray'), plt.show()

    return matchimg

def main(): 

    test = np.array([[0,0,0,0,8],[0,5,3,0,0],[0,3,0,4,0],[0,0,6,0,2],[0,0,0,0,0]])

    print(f'test:\n{test}')

    h,w = test.shape
    wsize = 3
    offset = wsize//2

    for y in range(offset, h-offset):
        for x in range(offset, w-offset):
            window = test[y-offset:y+offset+1, x-offset:x+offset+1]

            window[window < np.max(window)] = 1


    print(f'test:\n{test}')

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
    wSizeNMS = 3 # size of the window for NMS
    hThreshold = 400000 # threshold for defining what is a corner, if R > threshold

    print("Finding corners for img1...")
    corners_img1 = harrisNMS(img1, alpha, wSizeHarris, wSizeNMS, hThreshold)
    cv2.imwrite("test1.jpg", corners_img1)

    print("Finding corners for img2...")
    corners_img2 = harrisNMS(img2, alpha, wSizeHarris, wSizeNMS, hThreshold)
    cv2.imwrite("test2.jpg", corners_img2)

    return

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