# Computer Vision, Project 2
# Vanessa Hadlock, Stav Rones

import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import ndim
import imutils


##################################################
#################### COPRNERS ####################
##################################################

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

# use cv2.cornerHarris, threshold by 10%, cv2.cornerSubPix
def cvCorners(img):
        
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = np.float32(grey)

    # find Harris corners
    blockSize = 2 
    ksize = 3
    k = 0.04 
    corner_img = cv2.cornerHarris(grey, blockSize, ksize, k)

    cv2.imwrite("test.jpg", corner_img)

    #result is dilated for marking the corners
    corner_img = cv2.dilate(corner_img, None)

    cv2.imwrite("test_dilated.jpg", corner_img)

    # Threshold for an optimal value, it may vary depending on the image.
    thres = corner_img.max() * 0.01
    ret, corner_img = cv2.threshold(corner_img, thres, 255, 0)

    # img[corner_img > 0.01 * corner_img.max() ]= [0,0,255]

    cv2.imwrite("test_threshold.jpg", corner_img)

    # Conver the image back into 8 bit integers
    corner_img = np.uint8(corner_img)

    # define the criteria to stop and refine the corners
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(corner_img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    corners = cv2.cornerSubPix(corner_img, np.float32(centroids),(5,5),(-1,-1),criteria)

    # Now draw the corners onto the image 
    res = np.hstack((centroids,corners))
    res = np.int0(res)

    ret = np.zeros_like(grey)
    # img[res[:,1],res[:,0]]= [0,0,255]
    ret[res[:,3],res[:,2]] = 255

    return ret

##################################################
################# CORRESPONDENCE #################
##################################################

# uses cv2.ORB for feature detection, and cv2.BFMatcher for matching
def bfMatcher(img1, img2):

     ### Find Keypoints ###

    orb = cv2.ORB_create(nfeatures=500)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    orb_img1 = cv2.drawKeypoints(img1, kp1, None)
    cv2.imwrite("orb1.jpg", orb_img1)

    orb_img2 = cv2.drawKeypoints(img2, kp2, None)
    cv2.imwrite("orb2.jpg", orb_img2)

    # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # convert matches to src_pts and dst_pts
    srcMask = []
    dstMask = []
    for match in matches[0:50]:
        srcMask.append(match.queryIdx)
        dstMask.append(match.trainIdx)

    src_pts = cv2.KeyPoint_convert(kp1, srcMask)
    dst_pts = cv2.KeyPoint_convert(kp2, dstMask)

    # # draw first 50 matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)

    cv2.imwrite("cv_matches.jpg", match_img)

    return np.int16(np.hstack((src_pts, dst_pts)))


# Computes the SSD of two arrays
def SSD(f, g):
    return np.sum((f - g) ** 2)

# Computes the Cross Corellation of two arrays
def CC(f, g):
    f = np.int32(f)
    g = np.int32(g)

    return np.sum(f * g)

# Computes the Normalized Cross Corellation of two arrays
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

# Layers the corners on top of the orignial image in red
def addCornertoImage(img, cornerImg, outfilename):

    h, w, _ = img.shape
    for y in range(h):
        for x in range(w):

            if cornerImg[y,x] != 0:
                img[y,x] = [0,255,0]

    cv2.imwrite(outfilename, img)
    
    return

# Returns a list of elements [x1,y1,x2,y2] where (x1,y1) in img1 corresponds to (x2,y2) in img2
def correspondences(img1, img2, corners_img1, corners_img2, wsize, threshold, algo):

    # get the coordinates of the corners in img1

    if wsize % 2 == 0:
        wsize += 1
    offset = wsize // 2

    h1, w1 = corners_img1.shape
    h2, w2 = corners_img2.shape

    corners1 = []
    for y1 in range(offset, h1 - offset):
        for x1 in range(offset, w1 - offset):
            if corners_img1[y1][x1] == 255:
                corners1.append([y1, x1])

    corners2 = []          
    for y2 in range(offset, h2 - offset):
        for x2 in range(offset, w2 - offset):
            if corners_img2[y2][x2] == 255:
                corners2.append([y2, x2])

    # for each corner in img1, find the corner in img2 that is most similar

    img1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ret = []
    ncc = []
    for coord1 in corners1:

        nccList = []
        for coord2 in corners2:
            
            y1, x1 = coord1
            y2, x2 = coord2 

            w1 = img1[y1 - offset : y1 + offset + 1, x1 - offset : x1 + offset + 1]
            w2 = img2[y2 - offset : y2 + offset + 1, x2 - offset : x2 + offset + 1]

            if algo == "SSD":
                nccList.append(SSD(w1, w2))
            if algo == "CC":
                nccList.append(CC(w1, w2))
            if algo == "NCC":
                nccList.append(NCC(w1, w2))

        # print(f'nccList:\n{np.round(nccList, 5)}\n')

        if algo == "SSD":

            nccMin = min(nccList)

            if (nccMin < threshold) | (threshold == 0):
                pt2 = corners2[nccList.index(nccMin)]
                ret.append(coord1[::-1] + pt2[::-1])
                ncc.append(nccMin)
        else:
                
            nccMax = max(nccList)

            if nccMax > threshold:
                pt2: np.ndarray = corners2[nccList.index(nccMax)]
                ret.append(coord1[::-1] + pt2[::-1])
                ncc.append(nccMax)

    return ret

##################################################
#################### MATCHING ####################
##################################################

# compute homography for 
def drawMatches(img1, img2, correspondences):

    ###  get the 3x3 transformation homography ###
    dst_pts = []
    src_pts = []
    kp1 = []
    kp2 = []

    for row in correspondences:

            pt1 = [row[0], row[1]]
            pt2 = [row[2], row[3]]

            src_pts.append(pt1)
            dst_pts.append(pt2)

            kp1.append(cv2.KeyPoint(x=pt1[0], y=pt1[1], size=1))
            kp2.append(cv2.KeyPoint(x=pt2[0], y=pt2[1], size=1))

    print(src_pts)
            
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)   

    # Estimate the homography matrix using of four pairs each (collinear pairs 
    # are discarded) and a simple least-squares algorithm, and then compute the 
    # quality/goodness of the computed homography aka number of inliers\
    # Homography is then refined further (using inliers only in case of a 
    # robust method) with the Levenberg-Marquardt method to reduce the 
    # re-projection error even more

    correspondences = sorted(correspondences, key=lambda x: x[2])
    for row in correspondences:
        print(row)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    np.set_printoptions(suppress=True)
    print("Estimated homography : \n", H)

    ###  Draw matches between images ###

    print(len(kp1))

    goodMatches = []
    kp1_inliers = []
    kp2_inliers = []
    count = 0
    for i in range(len(correspondences)):
        if mask[i] == 1:
            kp1_inliers.append(kp1[i])
            kp2_inliers.append(kp2[i])
            goodMatches.append(cv2.DMatch(_imgIdx=0, _queryIdx=count, _trainIdx=count, _distance=0))
            count += 1

    matches_img = cv2.drawMatches(img1, kp1_inliers, img2, kp2_inliers, goodMatches, None)
    cv2.imwrite("matches.jpg", matches_img)
    

    return H

# Function takes in the two images that need to be warped together
# and the Homography generated from the 4 corresponding points in 
# the two images. The function determines the size of the output image,
# copies image 2 into the output image, and then warps image one into the
# output image using the homography matrix. The images are then blended
# using a feather technique.
# @params   img1, first image
#           img2, second image
#           H, the 3x3 homography
# @returns  out_img, the panorama output image
def warpimage(img1, img2, H, crop):

    # finding the size of the two images to determine the output image size when the
    # two images are warped together
    rows1, cols1 = img1.shape[:-1]
    rows2, cols2 = img2.shape[:-1]    

    # all the pixel coordinates of the reference image (img 1)
    list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)

    # all the pixel coordinates of the second reference image (img 2) that is going to be transformed
    temp_points = np.float32([[0,0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1,1,2)

    # calculate the transformation using the refined H matrix and the points in the 
    # second image to map where they need ot be in the first image
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

    # joining the two list of points from the img 1 points and the transformed img 2 points
    list_of_points = np.concatenate((list_of_points_2, list_of_points_1), axis=0)

    # defining width and the heigh of the ouput image using the number of pixels in
    # img 1 and img 2 
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    
    # defining the translation 
    translation_dist = [-x_min,-y_min]
    print("translation dist: ", translation_dist)

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    print("H translation: ", H_translation)

    # warp the first image onto the second image, using the transformation matrix
    # passing in the first image, the transformation matrix, and the width/height of the output image
    # this will place the first image into the background before adding in the second image
    output_img = cv2.warpPerspective(img1, H_translation.dot(H), (x_max-x_min, y_max-y_min))

    cv2.imwrite("out1.jpg", output_img)

    # gray1= cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    # gray1 = np.float32(gray1)

    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # gray2 = np.float32(gray2)

    # for i in range(gray1[translation_dist[1], rows1 + translation_dist[1]]):
    #     for j in range(gray1[translation_dist[0], rows1 + translation_dist[1]]):
    #         if gray1[i, j] == (255, 255, 255):
    #             dst = gray2[i, j]
    #         else:
    #             #alpha = 1.0/(i + 1)
    #             #beta = 1.0 - alpha
    #             dst[i, j] = cv2.addWeighted(gray1[i, j], 0.5, gray2[i, j], 0.5, 0.0)

    # output_img[translation_dist[1] : rows1 + translation_dist[1], translation_dist[0] : cols1 + translation_dist[0]] = (output_img[44:, 143:, :] / img2[:, :, :]) / 2
    output_img[translation_dist[1] : rows1 + translation_dist[1], translation_dist[0] : cols1 + translation_dist[0]] = img2

    cv2.imwrite("warpedimg.jpg", output_img)

    # Cropping an image 

    print("cropping...")

    if crop == 1:
        # cropping for the office 
        cropped_image = output_img[50:379, 7:648, 0:3]

    elif crop == 2:
        # cropping for the hallway
        cropped_image = output_img[22:362, 7:655, 0:3]

    return cropped_image


def main(): 

    ##################################################
    ######## Read in two images from each set ########
    ##################################################

    img1_filename = "DSC_0281"
    img2_filename = "DSC_0282"
    img3_filename = "DSC_0283"

    img4_filename = "DSC_0308"
    img5_filename = "DSC_0309"
    img6_filename = "DSC_0310"
    img7_filename = "DSC_0311"


    img1: np.ndarray = cv2.imread(f'DanaHallWay1/{img1_filename}.jpg') 
    img2: np.ndarray = cv2.imread(f'DanaHallWay1/{img2_filename}.jpg')
    img3: np.ndarray = cv2.imread(f'DanaHallWay1/{img3_filename}.jpg')
    img4: np.ndarray = cv2.imread(f'DanaOffice/{img4_filename}.jpg')
    img5: np.ndarray = cv2.imread(f'DanaOffice/{img5_filename}.jpg')
    img6: np.ndarray = cv2.imread(f'DanaOffice/{img6_filename}.jpg')
    img7: np.ndarray = cv2.imread(f'DanaOffice/{img7_filename}.jpg')


    # img1 = cv2.GaussianBlur(img1, (5,5), 0)
    # img2 = cv2.GaussianBlur(img2, (5,5), 0)
    # img3 = cv2.GaussianBlur(img3, (5,5), 0)
    # img4 = cv2.GaussianBlur(img4, (5,5), 0)
    # img5 = cv2.GaussianBlur(img5, (5,5), 0)
    # img6 = cv2.GaussianBlur(img6, (5,5), 0)
    # img7 = cv2.GaussianBlur(img7, (5,5), 0)

    ##################################################
    ### Detect corner pixels using Harris with NMS ###
    ##################################################

    alpha = 0.04 # constant between 0.04 - 0.06
    wSizeHarris = 3 # size of the window for the harris corner detection
    wSizeNMS = 3 # size of the window for NMS
    hThreshold = 400000 # threshold for defining what is a corner, if R > threshold

    print("Finding corners for img1...")
    corners_img1 = cvCorners(img1)
    cv2.imwrite(f'corners_img1.jpg', corners_img1)

    print("Finding corners for img2...")
    corners_img2 = cvCorners(img2)
    cv2.imwrite(f'corners_img2.jpg', corners_img2)

    print("Finding corners for img3...")
    corners_img3 = cvCorners(img3)
    cv2.imwrite(f'corners_img3.jpg', corners_img3)

    print("Finding corners for img4...")
    corners_img4 = cvCorners(img4)
    cv2.imwrite(f'corners_img4.jpg', corners_img4)

    print("Finding corners for img5...")
    corners_img5 = cvCorners(img5)
    cv2.imwrite(f'corners_img5.jpg', corners_img5)

    print("Finding corners for img6...")
    corners_img6 = cvCorners(img6)
    cv2.imwrite(f'corners_img6.jpg', corners_img6)

    print("Finding corners for img7...")
    corners_img7 = cvCorners(img7)
    cv2.imwrite(f'corners_img7.jpg', corners_img7)


    # print("Finding corners for img1...")
    # corners_img1 = harrisNMS(img1, alpha, wSizeHarris, wSizeNMS, hThreshold)
    # cv2.imwrite(f'corners_img1.jpg', corners_img1)
    # addCornertoImage(img1, corners_img1, f"{img1_filename}_corners.jpg")
    
    # print("Finding corners for img2...")
    # corners_img2 = harrisNMS(img2, alpha, wSizeHarris, wSizeNMS, hThreshold)
    # cv2.imwrite(f'corners_img2.jpg', corners_img2)
    # addCornertoImage(img2, corners_img2, f"{img2_filename}_corners.jpg")

    # print("Finding corners for img5...")
    # corners_img5 = harrisNMS(img5, alpha, wSizeHarris, wSizeNMS, hThreshold)
    # cv2.imwrite(f'corners_img5.jpg', corners_img5)
    # addCornertoImage(img5, corners_img5, f"{img5_filename}_corners.jpg")
    
    # print("Finding corners for img4...")
    # corners_img4 = harrisNMS(img4, alpha, wSizeHarris, wSizeNMS, hThreshold)
    # cv2.imwrite(f'corners_img4.jpg', corners_img4)
    # addCornertoImage(img4, corners_img4, f"{img4_filename}_corners.jpg")


    ##################################################
    ################ 2 Hallway Images ################
    ##################################################

    correspondences1 = bfMatcher(img1, img2)

    homography1 = drawMatches(img1, img2, correspondences1)

    hallway_crop = 2
    hallway = warpimage(img1, img2, homography1, hallway_crop)
    # cv2.imwrite("hallway_2warpedimg.jpg", hallway)

    ##################################################
    ################ 3 Hallway Images ################
    ##################################################

    # correspondences2 = bfMatcher(hallway, img3)
    # homography2 = drawMatches(hallway, img3, correspondences2)

    #result = warpimage(hallway, img3, homography2, hallway_crop)
    #cv2.imwrite("hallway_3warpedimg.jpg", result)


    ###################################################
    ################# 2 Office Images #################
    ###################################################

    correspondences3 = bfMatcher(img5, img4)

    homography3 = drawMatches(img5, img4, correspondences3)

    office_crop = 1
    office = warpimage(img5, img4, homography3, office_crop)

    cv2.imwrite("office_2warpedimg.jpg", office)

    # dst = cv2.addWeighted(office, 1, img2, 1, 0.0)
    
    # cv2.imwrite("office_blendingtest.jpg", office)


    ###################################################
    ################# 3 Office Images #################
    ###################################################

    # correspondences4 = bfMatcher(office, img6)

    # homography4 = drawMatches(office, img6, correspondences3)

    # office_crop = 1
    # office = warpimage(office, img6, homography4, office_crop)
    # cv2.imwrite("office_3warpedimg.jpg", office)

    # algo = "SSD"

    # if algo == "SSD":
    #     w = 7
    #     thres = 0
    # if algo == "CC":
    #     w = 7
    #     thres = 0
    # if algo == "NCC":
    #     w = 7
    #     thres = 0

    # print("Finding correspondences for set 1...")
    # correspondences1 = correspondences(img1, img2, corners_img1, corners_img2, w, thres, algo)
    # # print(f'correspondences1:\n{correspondences1}\n')
    
    # print("Finding correspondences for set 2...")
    # correspondences2 = correspondences(img1, img2, corners_img1, corners_img2, w, thres, algo)
    # # print(f'correspondences2:\n{correspondences2}\n')
       
    ##################################################
    ######## Estimate homography using RANSAC ########
    ##################################################



    ##################################################
    ############# Align images into one ##############
    ##################################################

    # print("Aligning images ...")
    # Takes in the two images and the correspondance list 
    # as well as true or false for RANSAC to determine if
    # we want to remove the outliers or not

    

    # alignImages(img1, img2, correspondences1, RANSAC)


    # Write aligned image to disk.
    # outFilename = "panorama.jpg"
    # print("Saving aligned image : ", outFilename)
    # cv2.imwrite(outFilename, panorama)

if __name__ == "__main__":
    main()