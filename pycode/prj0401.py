#!/usr/local/bin/python
from math import *
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import errno
import numpy as np
import cv2
from mpl_toolkits.mplot3d.axes3d import Axes3D

def visualize(gray_imgs,title=None):
    plt.figure()
    num = len(gray_imgs)
    for i in range(0,num):
        plt.subplot(100+10*num+i+1)
        plt.imshow(gray_imgs[i],cmap = 'gray')
        if not title == None:plt.title(title[i])
        plt.tight_layout()     
        plt.axis('off')

#plot figure
def plotFig(u_arr,x,y):
    fig = plt.figure()
    ax = Axes3D(fig)
    X, Y = np.meshgrid(np.arange(x), np.arange(y))
    ax.plot_surface(X, Y, u_arr, rstride=1, cstride=1, cmap='hot')


def main():
    image_dir = '/Users/gerrie/Documents/szu/course/DIP_assignments/Labimages/'
    img = mpimg.imread(os.path.join(image_dir,'Fig4.41(a).jpg'))
    img = img/255.0

    img_fft = np.fft.fft2(img)
    visualize([img,np.abs(img_fft)])
    # ##1.a 1.b 1.d
    [c_index_array,r_index_array] = np.meshgrid(range(img.shape[1]),range(img.shape[0]))
    ij_arr = (-1)**((r_index_array+c_index_array))
    imgb = img*(ij_arr)
    # plt.figure()
    # visualize([img,imgb])
    img_1b = np.fft.fft2(imgb)
    img_1f = np.log(1+np.abs(img_1b))
    img_1f_phase = np.angle(img_1b)
    # plotFig(img_1c,img_1c.shape[1],img_1c.shape[0])
    ##1.d 
    img_1d = np.fft.ifft2(img_1b)
    print img_1d.dtype
    img_1c = img_1d*ij_arr
    img_1e = np.real(img_1c)
    visualize([img,img_1f,img_1f_phase]
        ,['origin','frequency spectrum','Phase spectrum'])
    visualize([img,np.abs(img_1d),img_1e])

    ## 2.a 
    mask = np.zeros(img.shape,np.uint8)
    D_uv = (c_index_array - img.shape[1]/2)**2+(r_index_array - img.shape[0]/2)**2
    mask = np.exp(-D_uv /(2*20**2))
    plt.figure()
    plt.imshow(mask,cmap='gray')
    fimgb = np.fft.fft2(img)
    fimgc = np.fft.fftshift(fimgb)
    fimgd = fimgc * mask 
    imgz =  np.fft.ifft2(np.fft.ifftshift(fimgd))
    imgz = np.abs(imgz)
    imgz = (imgz - np.amin(imgz))/(np.amax(imgz)-np.amin(imgz))
    img_3a = np.abs(img - imgz)
    visualize([img,imgz,img_3a],['origin','low pass','sharpen'])
    plt.show()

if __name__ == '__main__':
    main()



# img_fft = np.fft.fft2(img)
# F1 = np.fft.fft2(img)
# # Now shift so that low spatial frequencies are in the center.
# F2 = np.fft.fftshift( F1 )
# # the 2D power spectrum is:
# img_fft = np.abs( F2 )**2
# visualize([img,img_fft])