import cv2
import numpy as np
from Config import cfg


def init_detect():
    pass


def image_detect(im_path):
    blobs, im_scales = _get_blobs(im_path)


if __name__ == '__main__':
    init_detect()
    im_list = []
    for im in im_list:
        image_detect(im)


def _get_blobs(im):
    blobs = {}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    return blobs, im_scale_factors


def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.MAX_SIZE:
            im_scale = float(cfg.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = _im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


def _im_list_to_blob(ims):
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    return blob
