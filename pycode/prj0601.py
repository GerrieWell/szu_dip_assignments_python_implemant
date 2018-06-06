#!/usr/bin/env python
# coding=utf-8
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from util import *

def gray2rbg(img,a,b):
	imgz = np.zeros(list(img.shape)+[3],dtype=img.dtype)
	imgz[:,:,0] = imgz[:,:,1] = imgz[:,:,2] = img
	pos = np.where(np.logical_and(img>=a , img<=b))
	imgz[pos[0],pos[1],0] = imgz[pos[0],pos[1],1] = 255
	imgz[pos[0],pos[1],2] = 0
	return imgz

def rgb_equalize_hist(src):
	dst = np.zeros_like(src)
	dst[:,:,0] = cv2.equalizeHist(src[:,:,0])
	dst[:,:,1] = cv2.equalizeHist(src[:,:,1])
	dst[:,:,2] = cv2.equalizeHist(src[:,:,2])
	return dst

def histeq2(im,cdf):
	imb = np.zeros_like(im)
	imb[:,:,0] = np.interp(im[:,:,0].flatten(),range(256),cdf).reshape(im.shape[:2])
	imb[:,:,1] = np.interp(im[:,:,1].flatten(),range(256),cdf).reshape(im.shape[:2])
	imb[:,:,2] = np.interp(im[:,:,2].flatten(),range(256),cdf).reshape(im.shape[:2])
	return imb

def main():
	plt.rcParams['figure.facecolor'] = 'white'
	file_path = '/Users/gerrie/Documents/szu/course/DIP_assignments/Labimages/'
	img = mpimg.imread(os.path.join(file_path,'Fig1.10(4).jpg'))
	##1.a 1.b
	imgb = gray2rbg(img,0,25)
	visualize([img,imgb],['original figure','rgb figure'])

	##2.a 2.c 
	imgc = mpimg.imread(os.path.join(file_path,'Fig6.35(5).jpg'))
	imgd = rgb_equalize_hist(imgc)
	imhistR,_ = np.histogram(imgd[:,:,0].flatten(),256,normed = True)
	imhistG,_ = np.histogram(imgd[:,:,1].flatten(),256,normed = True)
	imhistB,_ = np.histogram(imgd[:,:,2].flatten(),256,normed = True)
	imhist = (imhistR+imhistG+imhistB)/3
	cdf = imhist.cumsum()
	cdf = cdf*255/cdf[-1]
	imge = histeq2(imgd,cdf)
	fig = plt.figure();
	visualize([imgc,imgd,imge],['original figure','rgb equalized figure','average figure'])
	mpimg.imsave('./imgd.jpg',imgd)
	mpimg.imsave('./imge.jpg',imge)
	plt.show()

if __name__ == '__main__':
	main()