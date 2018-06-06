#!/usr/bin/env python
# coding=utf-8
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from util import *

def gaussian_noise(img,mean,var):
	noise =  mean + np.sqrt(var) * np.random.randn(img.shape[0],img.shape[1])
	return img+noise.astype(np.uint8)

def motion(img,T,a,b):
	imgb = (np.fft.fft2(img))
	umat,vmat = np.meshgrid(np.arange(img.shape[1]),np.arange(img.shape[0]))
	umat-=img.shape[1]/2
	vmat -= img.shape[0]/2
	tmp = np.exp( -1j * np.pi * umat*a+vmat*b)
	H = T/np.pi * np.divide(np.sin(np.pi * (umat*a + vmat*b) * tmp) , (umat*a+vmat*b) )
	imgc = H * imgb
	return np.fft.ifft((imgc)),H

def main():
	file_path = '/Users/gerrie/Documents/szu/course/DIP_assignments/Labimages/Fig5.26(a).jpg'
	img = mpimg.imread(file_path)
	# a,d
	dirty_img = gaussian_noise(img,0,10)
	motion_img,H = motion(img,1,0.1,0.1)
	motion_img = motion_img.astype(np.uint8)
	H = H.astype(np.uint8)
	print img.shape
	print motion_img.shape
	visualize([img,motion_img,H])



	plt.show()

if __name__ == '__main__':
	main()