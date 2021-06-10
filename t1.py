#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
  
            
def stitch_background(img1, img2, savepath=''):
    # "The output image should be saved in the savepath."
    # "Do NOT modify the code provided." 
     
     
    s = []
    d = []
    
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
    
    M, mask = cv2.findHomography( dst_pts,src_pts, cv2.RANSAC, 5.0)
    pts = np.float32([[0, 0], [0, img2_gray.shape[0]], [img2_gray.shape[1], img2_gray.shape[0]], [ img2_gray.shape[1], 0]])
    pts=pts.reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M) 
    # print(dst) 
    x = [0]
    y= [0]
    for p in range(len(dst)):
        x.append(int(dst[p][0][0]))
        y.append(int(dst[p][0][1])) 
    x1 =  -1 * (min(x))
    y1 =  -1 * (min(y))  
    # print(x1,y1)
    M[0][2] += x1
    M[1][2] += y1 
    im1 = img1
    im2 = img2
    x1 = img1.shape[1]+img2_gray.shape[1]
    y1 = img1.shape[0]+img2_gray.shape[0]
    x2 = im1.shape[1]+im2.shape[1]
    y2 = im1.shape[0]+im2.shape[0]
    # # stitching the image    
    new_img = cv2.warpPerspective(img1_gray, M, (x1,y1))
    new_img1= cv2.warpPerspective(im1, M, (x2,y2))
    
    patch=2
    
    x = [0]
    y= [0]
    for p in range(len(dst)):
        x.append(int(dst[p][0][0]))
        y.append(int(dst[p][0][1])) 
    x1 =  -1 * (min(x))
    y1 =  -1 * (min(y))  
    
    # print(y1) 
    for i in range(0,(img2_gray.shape[0]-patch),patch):
        for j in range(0,(img2_gray.shape[1]-patch),patch):
            
            patch1=img2_gray[i:i+patch,j:j+patch]
            patch2=new_img[i+y1:i+patch+y1,j:j+patch] 
            
            f = np.average(patch1)
            s = np.average(patch2) 
            if(f - s >= 0):
                new_img1[i+y1:i+patch+y1,j:j+patch]=im2[i:i+patch,j:j+patch]
                 
    for i in range(0,im2.shape[0]):
        for j in range(0,im2.shape[1]):
            if(img2_gray[i][j]!=0 and new_img[i+y1][j]==0):
                # print(img2[i][j]-new[i+y1][j])
                new_img1[i+y1][j]=im2[i][j]
     
    black = np.zeros(3)
    x_max = 0
    y_max = 0
    for i in range(new_img1.shape[0]):
        for j in range(new_img1.shape[1]):
            pixel_value = new_img1[i, j, :]
            if not np.array_equal(pixel_value, black):
                if j > x_max:
                    x_max = j
                if i > y_max:
                    y_max = i

    crop_img = new_img1[0:y_max,0:x_max, :]
    plt.imshow(crop_img)
    plt.show()
    cv2.imwrite(savepath,crop_img)
    return None
      
 
if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

