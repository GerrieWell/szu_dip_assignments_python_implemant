#!/usr/bin/env python
# coding=utf-8
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

def visualize(gray_imgs,title=None):
    num = len(gray_imgs)
    for i in range(0,num):
        plt.subplot(100+10*num+i+1)
        plt.imshow(gray_imgs[i],cmap = 'gray')
        if not title == None:
            plt.title(title[i])
        plt.tight_layout()     
        plt.axis('off')