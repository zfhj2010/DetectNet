import cv2
import numpy as np
import os.path as osp
import tensorflow as tf
import matplotlib.pyplot as plt
from Config import cfg
from MyNMS import zsoft_nms


def init_detect():
    model_path = osp.abspath(osp.join(osp.dirname(__file__), 'model'))
    model_file = osp.join(model_path, 'model.ckpt')
    if not osp.isfile(model_file + '.meta'):
        raise IOError('{:s} not found.\n'.format(model_file + '.meta'))
    with tf.device('/cpu:0'):
        tfconfig = tf.ConfigProto(log_device_placement=True)
        sess = tf.Session(config=tfconfig)
        saver = tf.train.import_meta_graph(model_file + '.meta')
        saver.restore(sess, model_file)
        def_gh = tf.get_default_graph()

    return sess, def_gh


def image_detect(env, graph, im):
    blobs, im_scales = _get_blobs(im)
    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    _, scores, bbox_pred, rois = _test_image(env, graph, blobs['data'], blobs['im_info'])

    boxes = rois[:, 1:5] / im_scales[0]
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])

    box_deltas = bbox_pred
    pred_boxes = _bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    cls_ind = 75
    cls_boxes = pred_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    keep = zsoft_nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    real_inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    if len(real_inds) == 0:
        return []
    else:
        real_boxes = [dets[real_ind] for real_ind in real_inds]

    _vis_detections(im, 'Meter', dets, CONF_THRESH)
    plt.show()
    return real_boxes


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


def _test_image(env, graph, image, im_info):
    image_op = graph.get_tensor_by_name('input_image:0')
    info_op = graph.get_tensor_by_name('input_im_info:0')
    score_op = graph.get_tensor_by_name('resnet_v1_101_5/cls_score/BiasAdd:0')
    prob_op = graph.get_tensor_by_name('resnet_v1_101_5/cls_prob:0')
    bbox_op = graph.get_tensor_by_name('add:0')
    rois_op = graph.get_tensor_by_name('resnet_v1_101_3/rois/concat:0')
    feed_dict = {image_op: image, info_op: im_info}
    cls_score, cls_prob, bbox_pred, rois = sess.run([score_op, prob_op, bbox_op, rois_op],
                                                    feed_dict=feed_dict)
    return cls_score, cls_prob, bbox_pred, rois


def _bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes


def _clip_boxes(boxes, im_shape):
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _vis_detections(im, class_name, dets, thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


if __name__ == '__main__':
    sess, def_graph = init_detect()
    im_list = ['01.jpg']
    for im_path in im_list:
        im = cv2.imread(im_path)
        image_detect(sess, def_graph, im)