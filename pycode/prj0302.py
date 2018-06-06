#!/usr/bin/env python
# coding=utf-8
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from scipy import signal

def noisy(image,salt_amount = 0.004,pepper_amount = 0.009):
    row,col = image.shape
    s_vs_p = 0.5
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(salt_amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in image.shape]
    out[coords] = 1
    # Pepper mode
    num_pepper = np.ceil(pepper_amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in image.shape]
    out[coords] = 0
    return out

def medfilt2(img):
    imgz = np.array(img)
    imgb = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT,value=0.)
    for i,j in np.ndindex(img.shape):
        tmp = imgb[i:i+3,j:j+3].flatten()
        imgz[i,j] = tmp[np.argsort(tmp)[4]]
    return imgz

def visualize(gray_imgs,title=None):
    num = len(gray_imgs)
    for i in range(0,num):
        plt.subplot(100+10*num+i+1)
        plt.imshow(gray_imgs[i],cmap = 'gray')
        if not title == None:plt.title(title[i])
        plt.tight_layout()     
        plt.axis('off')

def main():
    image_dir = '/Users/gerrie/Documents/szu/course/DIP_assignments/Labimages/'
    img01 = mpimg.imread(os.path.join(image_dir,'Fig1.10(4).jpg'))/255.0
    img02 = mpimg.imread(os.path.join(image_dir,'Fig1.10(5).jpg'))/255.0
    img0_a = img01 + img02
    img0_b = img01 - img02
    img0_c = img01 * img02
    img0_d = img01 / img02
    img0_e = img01 * 0.5
    visualize([img0_a,img0_b,img0_c,img0_d,img0_e,img01])

    ## 1.a
    img = mpimg.imread(os.path.join(image_dir,'Fig3.38(a).jpg'))
    img = img/255.0
    kernel = 1.0/9 * np.ones((3,3))
    img1a = signal.convolve2d(img,kernel,mode='same')
    plt.figure()
    visualize([img,img1a])

    ## 1.b 
    M37c = np.array(np.mat('0,-1,0 ; -1,4,-1; 0,-1,0'))
    M37d = np.array(np.mat('-1,-1,-1; -1,8,-1; -1,-1,-1'))

    maskM37c = img - signal.convolve2d(img,M37c,mode='same')
    maskM37d = img -  signal.convolve2d(img,M37d,mode='same')
    img1b_c = img + 1 * maskM37c
    img1b_d = img + 1 * maskM37d
    plt.figure()
    visualize([img,img1b_c,img1b_d])

    ## 1.c 
    M41d = np.array(np.mat('-1,-2,-1 ; 0,0,0; 1,2,1'))
    M41e = np.array(np.mat('-1,0,1; -2,0,2; -1,0,1'))
    maskM41d = img - signal.convolve2d(img,M41d,mode='same')
    maskM41e = img - signal.convolve2d(img,M41e,mode='same')
    img1c_d = img + 4.5 * maskM41d
    img1c_e = img + 4.5 * maskM41e
    plt.figure()
    visualize([img,img1c_d,img1c_e])

    ## 2.a b c d 
    plt.figure()
    img = mpimg.imread(os.path.join(image_dir,'Fig5.07(a).jpg'))
    img = img/255.0
    img2ac = noisy(img,0.2,0.2)
    img2b = medfilt2(img2ac)
    visualize([img,img2ac,img2b],['source image','salt pepper noise','med filter'])
    plt.show()


if __name__ == '__main__':
    main()