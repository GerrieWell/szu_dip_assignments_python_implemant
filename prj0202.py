#!/usr/bin/env python
# coding=utf-8

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

def reduce_gray_level(img,factor):
    scale = int(2**factor)
    img = (img/scale)*scale
    return img

def rescale(img,factor):
    size = [int(i*factor) for i in img.shape ]
    dest = np.zeros(size)
    for i,j in np.ndindex(dest.shape):
        dest[i,j] = img[int(i/factor),int(j/factor)]
    return dest

def bilinear_resize(img,m,n):  
    height,width =img.shape  
    emptyImage=np.zeros((m,n),img.dtype)  
    sh=float(m)/height  
    sw=float(n)/width  
    for i,j in np.ndindex((m,n)):
        x,y = i/sh,j/sw                     # 原图坐标: x: 纵坐标, y横坐标 . 
        p,q=(i+0.0)/sh-x , (j+0.0)/sw-y  
        x,y=min(int(x),height-2) , min(int(y),width-2)             #防止越界. 
        value = int(img[x,y]*(1-p)*(1-q)+img[x,y+1]*q*(1-p)+img[x+1,y]*(1-q)*p+img[x+1,y+1]*p*q)
        emptyImage[i, j] = value
    return emptyImage 

def visualize(gray_imgs,title=None):
    num = len(gray_imgs)
    for i in range(0,num):
        plt.subplot(100+10*num+i+1)
        plt.imshow(gray_imgs[i],cmap = 'gray')
        if not title == None:
            plt.title(title[i])
        plt.tight_layout()     

def main():
    image_dir = '/Users/gerrie/Documents/szu/course/DIP_assignments/Labimages/'
    test_mat = mpimg.imread(os.path.join(image_dir,'Fig2.21(a).jpg'))
    img219a = mpimg.imread(os.path.join(image_dir,'Fig2.19(a).jpg'))
    
    # # 1.a 1.b 
    dest128 = reduce_gray_level(test_mat,1)
    dest64 = reduce_gray_level(test_mat,2)
    dest32 = reduce_gray_level(test_mat,3)
    visualize([test_mat,dest128,dest64,dest32],['origin','128','64','32']) 

    # #2.b 2.c
    scale_256 = rescale(img219a,0.25)
    scale_1024 = rescale(scale_256,4)
    plt.figure()
    visualize([img219a,scale_256,scale_1024],['origin','256x256','1024x1024'])

    #3.a 3.b 3.c 
    bi_scale_256 = bilinear_resize(img219a,256,256)
    restore_img = bilinear_resize(bi_scale_256,1024,1024)
    plt.figure()
    visualize([img219a,bi_scale_256,restore_img])

    plt.show()

if __name__ == '__main__':
    main()