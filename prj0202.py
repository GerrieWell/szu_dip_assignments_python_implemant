#!/usr/bin/env python
# coding=utf-8

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

def reduce_gray_level(img,factor):
    """
    img: uint8 image
    """
    scale = int(2**factor)
    img = (img/scale)*scale
    # img = img.astype(np.uint8)
    return img

def rescale(img,factor):
    size = [int(i*factor) for i in img.shape ]
    dest = np.zeros(size)
    for i,j in np.ndindex(dest.shape):
        dest[i,j] = img[int(i/factor),int(j/factor)]
    return dest

def restore(img,factor):
    return img

def bilinear_resize(src,dstsize):#src size
    if src.ndim==3:  
        dstsize.append(3)  
    dst=np.array(np.zeros(dstsize),src.dtype)  
    factory=float(np.size(src,0))/dstsize[0]   
    factorx=float(np.size(src,1))/dstsize[1]  
    print 'factory',factory,'factorx',factorx  
    srcheight=np.size(src,0)  
    srcwidth=np.size(src,1)  
    print 'srcwidth',srcwidth,'srcheight',srcheight  
    for i in range(dstsize[0]):  
        for j in range(dstsize[1]):  
            y=float(i)*factory  
            x=float(j)*factorx  
            if y+1>srcheight: 
                y-=1  
            if x+1>srcwidth:  
                x-=1   
            cy=np.ceil(y)  
            fy=cy-1  
            cx=np.ceil(x)  
            fx=cx-1  
            w1=(cx-x)*(cy-y)  
            w2=(x-fx)*(cy-y)  
            w3=(cx-x)*(y-fy)  
            w4=(x-fx)*(y-fy)      
            if (x-np.floor(x)>1e-6 or y-np.floor(y)>1e-6):   
                t=src[fy,fx]*w1+src[fy,cx]*w2+src[cy,fx]*w3+src[cy,cx]*w4  
                t=np.ubyte(np.floor(t))  
                dst[i,j]=t  
                #print t  
            else:        
                dst[i,j]=(src[y,x])  
                # print src[i,j]  
    return dst  

def visualize(src,dest):
    plt.subplot(121)
    plt.imshow(src,cmap='gray')
    plt.subplot(122)
    plt.imshow(dest,cmap='gray')    

def main():
    image_dir = '/Users/gerrie/Documents/szu/course/DIP_assignments/Labimages/'
    test_mat = mpimg.imread(os.path.join(image_dir,'Fig2.22(a).jpg'))
    # test_mat = test_mat/255.0 
    # 1.a 1.b 
    dest = reduce_gray_level(test_mat,6)
    visualize(test_mat,dest) 

    # 2.a 2.b 
    dest = rescale(test_mat,2)
    figure()
    visualize(test_mat,dest)

    #2.b 2.c
    scale_256 = rescale(test_mat,0.25)
    scale_1024 = rescale(scale_256,4)
    figure()
    visualize(scale_256,scale_1024)

    #3.a 
    plt.show()

if __name__ == '__main__':
    main()