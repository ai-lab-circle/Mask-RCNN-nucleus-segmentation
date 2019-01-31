#author : Eric Yang
#content: generate mask image per each nuclei
#input: mask image
#output: generated mask images same as the number of nuclei

#modification: kimyy
#use dilation-eroision method to separate nucleus
#data:2019.01.05


import cv2
import os
import numpy as np
import imutils
import shutil
from PIL import Image


def load_image_into_numpy_array(image):
    # (im_width, im_height) = image.size
    (im_height,im_width, _) = image.shape
    # return   image.reshape(im_height, im_width, 3 ).astype(np.uint8)
    # (im_width, im_height, _) = image.shape
    # return   image.reshape(im_width, im_height, 3 )

    # return np.array(image.reshape( (im_height, im_width, 3)).astype(np.uint8) )
    return np.array(image.reshape( (im_height, im_width, 3)).astype(np.uint8) )

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

src_path = '../../datasets/nucleus/original_images/'


train_path = '../../datasets/nucleus/train_images/'
test_path = '../../datasets/nucleus/val_images/'


TEST_IMAGE_PATHS = os.listdir(src_path)

print(len(TEST_IMAGE_PATHS))


#kimyy
#read training/testing files

f = open("../../datasets/nucleus/train_w32_parent.txt", 'r')
image_list = f.readlines()

#define resize hight,weight
resize_im_height = 1024
resize_im_weight = 1024

#print(image_list)

for image_path in TEST_IMAGE_PATHS:
    if os.path.isdir(image_path): continue
    filename, file_extension = os.path.splitext(image_path)
    #print(filename, file_extension)
    print(image_path)
    if not (file_extension == '.jpg' or file_extension == '.JPG' or file_extension == '.png' or file_extension == '.PNG'): continue



    image = cv2.imread(src_path  +image_path, cv2.IMREAD_COLOR)


    #print(image.shape)
    # kimyy
    # separate data file into training and testing files
    parent_name = filename[:5]
    #print(parent_name)
    image_write_path = []
    if any(parent_name in s for s in image_list):
        print('training')
        image_write_path = train_path

    else:
        print('testing')
        image_write_path = test_path

    image_write_path += filename[:-4] + 'original'


    # make directory
    mask_folder_path = '/masks/'
    images_folder_path = '/images/'
    try:
        if not (os.path.isdir(image_write_path)):
            os.makedirs(os.path.join(image_write_path))
            os.makedirs(os.path.join(image_write_path + images_folder_path))
            os.makedirs(os.path.join(image_write_path + mask_folder_path))
    except OSError as e:
            print("Failed to create directory!!!!!")
            raise

    #kimyy

    #
    #shutil.copyfile(src_path + filename[:-4] + 'original.tif', image_write_path + images_folder_path+ filename[:-4] + 'original.jpg')
    # copy h&e image in either training or testing folder after resize that (1024x1024)
    tiff_im = Image.open(src_path + filename[:-4] + 'original.tif')
    tiff_im.thumbnail((resize_im_height, resize_im_weight), Image.ANTIALIAS)
    tiff_im.save(image_write_path + images_folder_path+ filename[:-4] + 'original.jpg', 'JPEG', quality=70)


    image_np = load_image_into_numpy_array(image)
    (im_height, im_width, c) = image_np.shape
    #print(im_height, im_width, c)
    div_x = im_width / 1000
    #half = int(im_width/2)
    # cv2.imshow('whole_page', cv2.resize(image_np, (int(im_width / div_x), int(im_height / div_x))))
    # cv2.waitKey(0)
    img2 = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('grey', img2)
    #cv2.imshow('grey', cv2.resize(img2, (int(im_width / div_x), int(im_height / div_x))))
    #cv2.waitKey(0)
    # canny = auto_canny(img2, sigma=0.01)
    canny =  cv2.Canny(img2, 0,255)
    # cv2.imshow('canny', canny)
    #cv2.imshow('canny', cv2.resize(canny, (int(im_width / div_x), int(im_height / div_x))))
    #cv2.waitKey(0)

    # cnts, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #kim
    # use erosion method to separte each object
    # 7x7 kernel ( if all pixel are one then result is one, otherwise zero)
    kernel = np.ones((7, 7), np.uint8)
    result = cv2.erode(img2, kernel, iterations=1)

    contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)

    # print(len(contours))
    # print(len(cnts))

    out_image = np.zeros((im_height, im_width), np.uint8)
    out_image[:] = 0
    # cv2.drawContours(out_image, contours, -1, (255), thickness=cv2.FILLED)
    cnts_num=0
    _cX = 0
    _cY = 0
    i = 0
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        if (M["m00"]!=0) :
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # if (i!=0 and cX==_cX and cY==_cY) : continue
            if (i!=0 and abs(cX-_cX)<=2 and abs(cY-_cY)<=2) : continue
            if (i==14 or i==15) : print(c)
            #print(i+1,'===============',cX, cY)
            # draw the contour and center of the shape on the image
            cv2.drawContours(out_image, [c], -1, (255, 255, 255), thickness=cv2.FILLED)
            #!! close/complete contours using convex hull
            #hull = cv2.convexHull(c, True)
            #cv2.drawContours(out_image, [c], 0, (255), -1)

            #cv2.drawContours(out_image, [hull], 0, (255), -1 )


            #cv2.circle(out_image, (cX, cY), 3, (255), -1)
            cnts_num=cnts_num+1
            i=i+1
            _cX = cX
            _cY = cY

            #kimkyy
            #make mask image each object independenly
            indv_image = np.zeros((im_height, im_width), np.uint8)
            indv_image[:] = 0
            cv2.drawContours(indv_image, [c], -1, (255, 255, 255), thickness=cv2.FILLED)

            # use dilation method to recover original shape
            indv_image = cv2.dilate(indv_image, kernel, iterations=1)
            contour = cv2.findContours(indv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #contour = imutils.grab_contours(contour)



            # erosion-dilation(=~ original)
            out_image2 = cv2.dilate(out_image, kernel, iterations=1)
            cv2.drawContours(out_image2, [c], -1, (255, 255, 255), thickness=cv2.FILLED)



            #cv2.imshow('erosion', cv2.resize(out_image, (int(im_width / div_x), int(im_height / div_x))))
            #cv2.imshow('erosion-dliation(=~ original)', cv2.resize(out_image2, (int(im_width / div_x), int(im_height / div_x))))

            #cv2.imshow('(indivitual) erosion-dliation', cv2.resize(indv_image, (int(im_width / div_x), int(im_height / div_x))))

            #kimyy
            #write mask image indepentently
            #print(image_write_path + mask_folder_path + filename + '_' + str(i) )

            #print(image.shape)
            # rezie image 1024x1024
            #cv2.waitKey(0)
            indv_image = cv2.resize(indv_image, (resize_im_height, resize_im_weight), interpolation=cv2.INTER_AREA)

            cv2.imwrite(image_write_path + mask_folder_path + filename + '_' + str(i) + '.png', indv_image)


            #cv2.waitKey(0)


    #print(cnts_num)
    # kernel = np.ones((3, 3), np.uint8)
    # out_image = cv2.morphologyEx(out_image, cv2.MORPH_CLOSE, kernel, iterations=6)
    # cv2.imshow('out', out_image)
    # cv2.imwrite(des_path + '/' + filename + '_' + str(i) + file_extension, out_image)
    #cv2.imshow('out', cv2.resize(out_image, (int(im_width / div_x), int(im_height / div_x))))
    # cv2.imshow('out', out_image)
    #cv2.waitKey(0)


f.close()
"""
    for i in range(len(contours)):
        out_image = np.zeros((im_height, im_width), np.uint8)
        out_image[:] = 0
        cv2.drawContours(out_image, contours,i,(255),thickness=cv2.FILLED)
        kernel = np.ones((3, 3), np.uint8)
        out_image = cv2.morphologyEx(out_image, cv2.MORPH_CLOSE, kernel, iterations=15)
        # cv2.imshow('out', out_image)
        cv2.imwrite(des_path + '/'  + filename+'_'+str(i)+file_extension, out_image)
        # cv2.imshow('out', cv2.resize(out_image, (int(im_width / div_x), int(im_height / div_x))))
        # cv2.waitKey(0)
"""
