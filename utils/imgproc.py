import cv2 as cv
import numpy as np

def crop_img(img, stride=2, size=[128, 128]):
    max_rows, max_cols = img.shape[:2]
    ret = []

    for c in range(0, max_cols, stride):
        if (c + size[1]) >= max_cols:
            break
        for r in range(0, max_rows, stride):
            if (r + size[0]) >= max_rows:
                break
            ret.append(img[r:r+size[0], c:c+size[1]])

    return ret

def int_to_float(int_img):
    float_img = int_img

    if float_img.dtype == np.uint8:
        float_img = float_img.astype(np.float32)
        float_img /= 255

    return float_img

def to_gray(img):
    if len(img.shape) != 3:
        raise ValueError("Image must have 3 dimension, but it has {0}".format(len(img.shape)))

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_img = int_to_float(gray_img) 

    return gray_img

def read_img(filename_img, filename_mask=None):
    """
    Function used to abstract the reading process of an image,
    it returns [img, mask] when filename_mask is passed.
    Otherwise, it returns [img]
    """
    img = cv.imread(filename_img)

    if filename_mask is not None:
        return [img, cv.imread(filename_mask)]

    return [img]

def copy_border(image, padding, value=[255, 255, 255]):
    print('copy_border')
    return cv.copyMakeBorder(image, padding, padding, padding, padding, cv.BORDER_CONSTANT, value=value)

def remove_border(image, padding):
    print('remove_border')
    
    width = image.shape[0]
    height = image.shape[1]

    new_width = width - padding
    new_height = height - padding

    assert (new_width > 0), "new_width should be bigger than zero!"
    assert (new_height > 0), "new_height should be bigger than zero!"

    assert (new_width > padding), "new_width should be bigger than padding!"
    assert (new_height > padding), "new_width should be bigger than padding!"

    return image[padding:new_width, padding:new_height]


def write_img(filename_img, img):
    """
    Function used to abstract the writing process of an image
    """
    if img.dtype == np.float:
        img *= 255
    cv.imwrite(filename_img, img)
