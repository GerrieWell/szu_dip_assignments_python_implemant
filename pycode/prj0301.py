#!/usr/bin/env python
# coding=utf-8
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

def convlog(ima,c):
    imab = ima + np.ones(ima.shape) 
    imc = c * np.log(imab)
    return  (imc/np.log(256) * 255).astype(np.uint8)

def conv_exp(ima,gamma,c):
  imac= ima/255.0
  imad = c*(imac**gamma)
  return (imad * 255).astype(np.uint8)

def im_hist(ima,bin_size = 256):
    patch = 256.0/bin_size
    im_bin = (np.floor(ima/patch)* patch).astype(np.uint8)
    unique,count = np.unique(im_bin , return_counts = True)
    histogram = np.zeros([bin_size])   #+1
    for u, c in zip(unique, count):
        histogram[u] = c
    return np.array(histogram)

def equalize_hist(ima):
    hist = im_hist(ima,256)
    hist = hist / float(np.sum(hist))
    dest_hist = 255 * hist.cumsum() 
    result = np.interp(ima.flatten(),range(len(dest_hist)),dest_hist[:])
    return result.reshape(ima.shape).astype(np.uint8) , dest_hist

def visualize(gray_imgs,title=None):
    num = len(gray_imgs)
    for i in range(0,num):
        plt.subplot(100+10*num+i+1)
        plt.imshow(gray_imgs[i],cmap = 'gray')
        if not title == None:
            plt.title(title[i])
        plt.tight_layout()     
        plt.axis('off')

def main():
    image_dir = '/Users/gerrie/Documents/szu/course/DIP_assignments/Labimages/'
    img = mpimg.imread(os.path.join(image_dir,'Fig3.08(a).jpg'))

    ## prj0301-1.a 1.b 
    dest1a = convlog(img,0.8)
    dest1b = conv_exp(img,0.35,0.7)
    visualize([img,dest1a,dest1b],['source','log','exponential'])
    plt.figure()
    dest1b_1 = conv_exp(img,0.35,0.2)
    visualize([dest1b,dest1b_1])
    plt.figure()
    dc1 = conv_exp(img,0.2,0.7)
    dc2 = conv_exp(img,0.8,0.7)
    visualize([dc1,dc2])
    ## prj0303 - 2.a 2.b 2.c 
    # _,axs = plt.subplots(1,2)
    # hist = im_hist(img,256)
    # axs[0].imshow(img,cmap='gray')
    # axs[1].bar(np.arange(len(hist)),hist,0.35,color='blue')

    # dest_2b,dest_hist = equalize_hist(img)
    # _,axs = plt.subplots(1,2)
    # axs[0].bar(np.arange(len(hist)),hist,0.5,color='blue')
    # axs[1].bar(np.arange(len(dest_hist)),dest_hist,0.5,color='red')

    # dest_hist2 = im_hist(dest_2b,256)
    # _,axs = plt.subplots(1,2)
    # axs[0].imshow(dest_2b,cmap='gray')
    # axs[1].bar(np.arange(len(dest_hist2)),dest_hist2,0.35,)
    plt.show()

if __name__ == '__main__':
    main()



def histeq(img,nbr_bins=255):
    """ Histogram equalization of a grayscale image. """
    
    # 获取直方图p(r)
    imhist, bins = np.histogram(img.flatten(), nbr_bins, normed = True)
    
    # 获取T(r)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf /cdf[-1] 
    
    # 获取s，并用s替换原始图像对应的灰度值
    result = np.interp(img.flatten(),bins[:-1],cdf)
    
    return result.reshape(img.shape),cdf