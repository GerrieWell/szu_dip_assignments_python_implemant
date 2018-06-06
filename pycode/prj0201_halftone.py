#!/usr/bin/env python
# coding=utf-8
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

def __gray_template_genarate(placehold):
    m = np.zeros(9).astype(np.uint8)
    grays_list = [np.array(m).reshape(3,3)] # yield m
    for p in placehold:
        m[p] = 255
        grays_list.append(np.array(m).reshape(3,3))#yield m.reshape((3,3))
    return grays_list

grays_mat = __gray_template_genarate([1,8,0,6,2,5,7,3,4])

def halftone(src):
    ten_gray = np.floor(src /26)
    dest = np.zeros([i*3 for i in ten_gray.shape ]).astype(np.uint8)
    for i,j in np.ndindex(ten_gray.shape):
        dest[3*i:3*(i+1),3*j:3*(j+1)] = grays_mat[int(ten_gray[i,j])]
    return dest

def resizeA4(img):
    dpi = 720
    max_pixel_height = int(min(29.7* 0.393700787 * dpi, img.shape[0]))
    max_pixel_width  = int(min(21.0* 0.393700787 * dpi, img.shape[1]))
    img = cv2.resize(img,(max_pixel_height,max_pixel_width))
    return img

def test_shade_img():
    img = np.zeros((256,256))
    for c in range(256):
        img[:,c] = c
    return img.astype(int)

def visualize(src,dest):
    plt.subplot(121)
    plt.imshow(src,cmap='gray')
    plt.subplot(122)
    plt.imshow(dest,cmap='gray')       
    plt.tight_layout()     

def main():
    ## prj0201_a
    image_dir = '/Users/gerrie/Documents/szu/course/DIP_assignments/Labimages/'
    test_mat = mpimg.imread(os.path.join(image_dir,'Fig2.22(a).jpg'))
    dest =  halftone(test_mat)
    dest = resizeA4(dest)
    visualize(test_mat,dest)

    ## prj0201_b
    plt.figure()
    shade_img = test_shade_img()
    dest =  halftone(shade_img)
    visualize(shade_img,dest)

    ## prj0201_c
    for im in ['Fig2.22(a).jpg','Fig2.22(b).jpg','Fig2.22(c).jpg']:
        plt.figure()
        test_mat = mpimg.imread(os.path.join(image_dir,im))
        dest =  halftone(test_mat)
        dest = resizeA4(dest)
        visualize(test_mat,dest)
    plt.show()

if __name__ == '__main__':
    main()