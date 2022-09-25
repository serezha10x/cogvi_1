import cv2
import numpy as np
from matplotlib import pyplot as plt

ENTER_CODE = 13
ESC_CODE = 27
SPACE_CODE = 32
CROP_STEP = 0.1
IMAGE_PATHS = ['/home/serezha/Documents/univer/COGVI/1.jpg',
               '/home/serezha/Documents/univer/COGVI/2.jpg',
               '/home/serezha/Documents/univer/COGVI/3.jpg',
               '/home/serezha/Documents/univer/COGVI/4.jpg',
               '/home/serezha/Documents/univer/COGVI/5.jpg']


def _linear_contrast(img):
    alpha = 0.8
    beta = 10
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    cv2.imshow('Adjusted Image', adjusted_img)
    return adjusted_img


def _convert_image_to_luv(img):
    luv_img = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
    cv2.imshow("LUV Color Scheme Image", luv_img)
    return luv_img


def _convert_image_to_grey(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grey Image', gray_img)
    return gray_img


def _median_filter(img):
    median = cv2.medianBlur(src=img, ksize=5)
    cv2.imshow('Median Blurred', median)
    return median


def _gauss_filter(img):
    gaussian_blur = cv2.GaussianBlur(src=img, ksize=(9, 9), sigmaX=0, sigmaY=0)
    cv2.imshow('Gaussian Blurred', gaussian_blur)
    return gaussian_blur


def _custom_filter(img):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    custom_filter_img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    cv2.imshow('Custom Filter Image', custom_filter_img)
    return custom_filter_img


if __name__ == '__main__':
    for image_path in IMAGE_PATHS:
        img = cv2.imread(image_path)

        color = ('b', 'g', 'r')
        for channel, col in enumerate(color):
            histr = cv2.calcHist([img], [channel], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.title('Histogram for color scale picture of original image')
        plt.show()

        key = cv2.waitKey()
        luv_img = _convert_image_to_luv(img)
        key = cv2.waitKey()
        plt.hist(luv_img.ravel(), 256, [0, 256], label='LUV Color Scheme Plot')
        plt.show()

        key = cv2.waitKey()
        grey_img = _convert_image_to_grey(img)
        plt.hist(grey_img.ravel(), 256, [0, 256], label='Grey Plot')
        plt.show()

        key = cv2.waitKey()
        adjusted_img = _linear_contrast(img)
        plt.hist(adjusted_img.ravel(), 256, [0, 256], label='Adjusted Plot')
        plt.show()

        key = cv2.waitKey()
        median = _median_filter(img)
        plt.hist(median.ravel(), 256, [0, 256], label='Median Blurred Plot')
        plt.show()

        key = cv2.waitKey()
        gaussian_blur = _gauss_filter(img)
        plt.hist(gaussian_blur.ravel(), 256, [0, 256], label='Gaussian Blurred Plot')
        plt.show()

        key = cv2.waitKey()
        custom_filter_img = _custom_filter(img)
        plt.hist(custom_filter_img.ravel(), 256, [0, 256], label='Custom Filter Plot')
        plt.show()

        while True:
            height = img.shape[0]
            width = img.shape[1]
            print("Height: " + str(height))
            print("Wide: " + str(width))
            cv2.imshow('Original Image', img)

            key = cv2.waitKey()

            if key == SPACE_CODE:
                height_step = int(height * CROP_STEP)
                width_step = int(width * CROP_STEP)
                print("Height Step: " + str(height_step))
                print("Width Step: " + str(width_step))

                img = img[height_step:height - height_step, width_step:width - width_step]
            if key == ENTER_CODE:
                break
            if key == ESC_CODE:
                cv2.destroyAllWindows()
                exit(0)
