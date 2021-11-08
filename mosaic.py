# Computer Vision, Project 2
# Vanessa Hadlock, Stav Rones

import numpy as np
import cv2

##################################################
##################### CORNERS ####################
##################################################

# use cv2.cornerHarris, threshold by 10%, cv2.cornerSubPix
def cvCorners(img):
    
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = np.float32(grey)

    # Find corners using harris corner detector
    blockSize = 2 
    ksize = 3
    k = 0.04 
    corner_img = cv2.cornerHarris(grey, blockSize, ksize, k)

    # Dilate and threshold the image by 10%
    thres = corner_img.max() * 0.01
    corner_img = cv2.dilate(corner_img, None)
    ret, corner_img = cv2.threshold(corner_img, thres, 255, 0)
    
    # use cv2.cornerSubPix to refine the corners
    # This essitially performs NMS
    corner_img = np.uint8(corner_img)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(corner_img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(corner_img, np.float32(centroids),(5,5),(-1,-1),criteria)

    # Create corner image
    res = np.hstack((centroids,corners))
    res = np.int0(res)

    ret = np.zeros_like(grey)
    ret[res[:,3],res[:,2]] = 255

    return ret

##################################################
################# CORRESPONDENCE #################
##################################################

# Computes the Normalized Cross Corellation of two 2D arrays
def NCC(f, g):

    f = np.int32(f)
    g = np.int32(g)
    
    fmag = np.sum(f ** 2) ** (1/2)
    gmag = np.sum(g ** 2) ** (1/2)

    return np.sum((f / fmag) * (g / gmag))

# computer ncc for two colored images
def NCCColor(f, g):

    nccBand = []
    for i in range(3):
        nccBand.append(NCC(f[:,:,i],g[:,:,i]))
    return np.average(nccBand)

# Layers the corners on top of the orignial image in red
def addCornertoImage(img, cornerImg, outfilename):

    h, w, _ = img.shape
    for y in range(h):
        for x in range(w):

            if cornerImg[y,x] != 0:
                img[y,x] = [0,0,255]

    cv2.imwrite(outfilename, img)
    
    return

# Returns a list of elements [x1,y1,x2,y2] where (x1,y1) in img1 corresponds to (x2,y2) in img2
def findCorrespondences(img1, img2, corners_img1, corners_img2, wsize, threshold):

    # get the coordinates of the corners in img1

    if wsize % 2 == 0:
        wsize += 1
    offset = wsize // 2

    h1, w1 = corners_img1.shape
    h2, w2 = corners_img2.shape

    corners1 = []
    for y1 in range(offset, h1 - offset):
        for x1 in range(offset, w1 - offset):
            if corners_img1[y1][x1] > 0:
                corners1.append([y1, x1])

    corners2 = []          
    for y2 in range(offset, h2 - offset):
        for x2 in range(offset, w2 - offset):
            if corners_img2[y2][x2] > 0:
                corners2.append([y2, x2])

    # for each corner in img1, find the corner in img2 that is most similar

    ret = []
    for coord1 in corners1:

        # nccList = []
        nccMax = 0
        nccMax_idx = 0
        for i in range(len(corners2)):

            coord2 = corners2[i]
            
            y1, x1 = coord1
            y2, x2 = coord2 

            w1 = img1[y1 - offset : y1 + offset + 1, x1 - offset : x1 + offset + 1, :]
            w2 = img2[y2 - offset : y2 + offset + 1, x2 - offset : x2 + offset + 1, :]

            ncc = NCCColor(w1, w2)
            if(ncc > nccMax):
                nccMax = ncc
                nccMax_idx = i
    
        if nccMax > threshold:
            pt2: np.ndarray = corners2[nccMax_idx]
            ret.append(coord1[::-1] + pt2[::-1])

    return ret

# Draws the corresponding features of two images
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
            
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)   

    ### Refine correspondences ###

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    ###  Draw matches between images ###

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

    return matches_img

##################################################
#################### Mosaicing ###################
##################################################

# compute homography for the given correspondences
def findHomgraphy(correspondences):

    ###  get the 3x3 transformation homography ###
    dst_pts = []
    src_pts = []

    for row in correspondences:

            pt1 = [row[0], row[1]]
            pt2 = [row[2], row[3]]

            src_pts.append(pt1)
            dst_pts.append(pt2)
            
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)   

    # Estimate the homography matrix using of four pairs each (collinear pairs 
    # are discarded) and a simple least-squares algorithm, and then compute the 
    # quality/goodness of the computed homography aka number of inliers\
    # Homography is then refined further (using inliers only in case of a 
    # robust method) with the Levenberg-Marquardt method to reduce the 
    # re-projection error even more

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    return H

def warpimage(img1, img2, H):

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

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # warp the first image onto the second image, using the transformation matrix
    # passing in the first image, the transformation matrix, and the width/height of the output image
    output_img = cv2.warpPerspective(img1, H_translation.dot(H), (x_max-x_min, y_max-y_min))

    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img2

    return output_img

##################################################
#################### MAIN ####################
##################################################

def main(): 

    ##################################################
    ######## Read in two images from each set ########
    ##################################################

    img1_filename = "DSC_0281"
    img2_filename = "DSC_0282"
    img3_filename = "DSC_0308"
    img4_filename = "DSC_0309"

    img1: np.ndarray = cv2.imread(f'DanaHallWay1/{img1_filename}.jpg') 
    img2: np.ndarray = cv2.imread(f'DanaHallWay1/{img2_filename}.jpg')
    img3: np.ndarray = cv2.imread(f'DanaOffice/{img3_filename}.jpg')
    img4: np.ndarray = cv2.imread(f'DanaOffice/{img4_filename}.jpg')

    ##################################################
    ### Detect corner pixels using Harris with NMS ###
    ##################################################

    print('Finding corners for img1...')
    corners_img1 = cvCorners(img1)
    addCornertoImage(img1,corners_img1,f"{img1_filename}_corners.jpg")

    print('Finding corners for img2...')
    corners_img2 = cvCorners(img2)
    addCornertoImage(img2,corners_img2,f"{img2_filename}_corners.jpg")

    print('Finding corners for img3...')
    corners_img3 = cvCorners(img3)
    addCornertoImage(img3,corners_img3,f"{img3_filename}_corners.jpg")

    print('Finding corners for img4...')
    corners_img4 = cvCorners(img4)
    addCornertoImage(img4,corners_img4,f"{img4_filename}_corners.jpg")
    
    ##################################################
    ############## Find correspondences ##############
    ##################################################

    wNcc = 7
    thres = 0

    print("Finding correspondences for set 1...")
    correspondences1 = findCorrespondences(img1, img2, corners_img1, corners_img2, wNcc, thres)

    print("Finding correspondences for set 2...")
    correspondences2 = findCorrespondences(img3, img4, corners_img3, corners_img4, wNcc, thres)

    ##################################################
    ################## Draw Matches ##################
    ##################################################

    print("Drawing matches for set 1...")
    matches1 = drawMatches(img1, img2, correspondences1)
    cv2.imwrite("matches1.jpg", matches1)

    print("Drawing matches for set 2...")
    drawMatches(img3, img4, correspondences2)
    cv2.imwrite("matches2.jpg", matches1)
       
    ##################################################
    ######## Estimate homography using RANSAC ########
    ##################################################

    print("Fiding homograpby for set 1...")
    H1 = findHomgraphy(correspondences1)

    print("Fiding homograpby for set 2...")
    H2 = findHomgraphy(correspondences2)

    ##################################################
    ############# Align images into one ##############
    ##################################################

    print("Warping images for set 1...")
    warpedImg1 = warpimage(img1, img2, H1)
    cv2.imwrite("warpedimg1.jpg", warpedImg1)

    print("Warping images for set 2...")
    warpedImg2 = warpimage(img3, img4, H2)
    cv2.imwrite("warpedimg2.jpg", warpedImg2)

if __name__ == "__main__":
    main()