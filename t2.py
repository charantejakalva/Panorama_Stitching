# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

def stitch_twoimgs(img1,img2): 
    s = []
    d = []
    
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show() 
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
    sift = cv2.SIFT_create()
 
    # Using SIFT to extract key points and descriptors
    kp_img1, des_img1 = sift.detectAndCompute(img1_gray, None)
    kp_img2, des_img2 = sift.detectAndCompute(img2_gray, None)
    
    
    for j1 in range(des_img2.shape[0]):
        dist=np.zeros(des_img1.shape[0])
        for i1 in range(des_img1.shape[0]):
            dist[i1]= cv2.norm(des_img2[j1]-des_img1[i1],cv2.NORM_L2)
         
        (f1,f2)=np.partition(dist,2)[0:2]  
        if f1/f2<0.8: 
            s.append(j1)
            d.append(np.where(dist==f1)[0][0])   
        
    src_pts = []  
    dst_pts = []
    for q in s:
        src_pts.append( kp_img2[q].pt)
    for q in d:
        dst_pts.append(kp_img1[q].pt)
    
    src_pts = np.float32(src_pts).reshape(-1,1,2)
    dst_pts = np.float32(dst_pts).reshape(-1,1,2) 
    # find homography
    
    M, mask = cv2.findHomography( src_pts,dst_pts, cv2.RANSAC, 5.0)
    pts = np.float32([[0, 0], [0, img2_gray.shape[0]], [img2_gray.shape[1], img2_gray.shape[0]], [ img2_gray.shape[1], 0]])
    pts=pts.reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    img1_gray = cv2.polylines(img1_gray, [np.int32(dst)], True, 255, 200, cv2.LINE_AA)
    # print(dst)
    x = [0]
    y= [0]
    for p in range(len(dst)):
        x.append(int(dst[p][0][0]))
        y.append(int(dst[p][0][1])) 
    x1 =  -1 * (min(x))
    y1 =  -1 * (min(y))  
    M[0][2] += x1
    M[1][2] += y1
    
    x2= max(x) + x1
    y2= max(y) + y1 
    x3=int(max(x2,x1+img1.shape[1]))
    y3=int(max(y2,y1+img1.shape[0])) 
    
    # # stitching the image    
    new_img = cv2.warpPerspective(img2, M, (x3,y3))
    new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    plt.imshow(new_gray)
    plt.show()
    
    new_img[y1: y1 + img1.shape[0], x1: x1 + img1.shape[1]] = img1
    
     
    black_pixel = np.zeros(3)
    x_max = 0
    y_max = 0
    x_min = new_img.shape[1]
    y_min = new_img.shape[0]
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            pixel_value = new_img[i, j, :]
            if not np.array_equal(pixel_value, black_pixel):
                if j > x_max:
                    x_max = j
                if i > y_max:
                    y_max = i
                if j < x_min:
                    x_min = j
                if i < y_min:
                    y_min = i
    crop_img = new_img[y_min:y_max,x_min:x_max, :]
    plt.imshow(crop_img)
    plt.show()
    return crop_img


def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    n = len(imgs)
    mat = np.zeros((n,n))
    for i in range(len(imgs)):
        img1 = imgs[i]
        for j in range(len(imgs)):
            img2 = imgs[j]
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()

            # finding the key points and descriptors with SIFT
            kp_img1, des_img1 = sift.detectAndCompute(img1_gray, None)
            kp_img2, des_img2 = sift.detectAndCompute(img2_gray, None) 
            k = 0
            for j1 in range(des_img2.shape[0]):
                dist=np.zeros(des_img1.shape[0])
                for i1 in range(des_img1.shape[0]):
                    dist[i1]= cv2.norm(des_img2[j1]-des_img1[i1],cv2.NORM_L2)
                 
                (f1,f2)=np.partition(dist,2)[0:2]  
                if f1/f2<0.8: 
                    k=k+1  
            mat_ratio = max(k/des_img1.shape[0], k/des_img2.shape[0])
            # print(k,des_img1.shape[0],des_img2.shape[0])
            if mat_ratio >= 0.2:
                mat[i][j] = 1 
    # print(np.int32(mat)) 
    n_imgs = [] 
    for i in range(len(imgs)): 
        n_imgs.append(imgs[i]) 
            
    pan_img = n_imgs[0] 
    for i in range(1,len(n_imgs)): 
        pan_img = stitch_twoimgs(n_imgs[i],pan_img) 
    plt.imshow(pan_img) 
    cv2.imwrite(savepath,pan_img) 

    return np.int32(mat)

if __name__ == "__main__":
    #task2
    
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
     
    
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    # bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    print(overlap_arr2)
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
