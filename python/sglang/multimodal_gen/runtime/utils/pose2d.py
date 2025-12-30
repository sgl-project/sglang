# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import os
import random
import warnings
from typing import List

import cv2
import numpy as np
from PIL import Image


def get_mask_boxes(mask):
    """

    Args:
        mask: [h, w]
    Returns:

    """
    y_coords, x_coords = np.nonzero(mask)
    x_min = x_coords.min()
    x_max = x_coords.max()
    y_min = y_coords.min()
    y_max = y_coords.max()
    bbox = np.array([x_min, y_min, x_max, y_max]).astype(np.int32)
    return bbox


def get_aug_mask(body_mask, w_len=10, h_len=20):
    body_bbox = get_mask_boxes(body_mask)

    bbox_wh = body_bbox[2:4] - body_bbox[0:2]
    w_slice = np.int32(bbox_wh[0] / w_len)
    h_slice = np.int32(bbox_wh[1] / h_len)

    for each_w in range(body_bbox[0], body_bbox[2], w_slice):
        w_start = min(each_w, body_bbox[2])
        w_end = min((each_w + w_slice), body_bbox[2])
        # print(w_start, w_end)
        for each_h in range(body_bbox[1], body_bbox[3], h_slice):
            h_start = min(each_h, body_bbox[3])
            h_end = min((each_h + h_slice), body_bbox[3])
            if body_mask[h_start:h_end, w_start:w_end].sum() > 0:
                body_mask[h_start:h_end, w_start:w_end] = 1

    return body_mask


def get_mask_body_img(img_copy, hand_mask, k=7, iterations=1):
    kernel = np.ones((k, k), np.uint8)
    dilation = cv2.dilate(hand_mask, kernel, iterations=iterations)
    mask_hand_img = img_copy * (1 - dilation[:, :, None])

    return mask_hand_img, dilation


def get_face_bboxes(kp2ds, scale, image_shape, ratio_aug):
    h, w = image_shape
    kp2ds_face = kp2ds.copy()[23:91, :2]

    min_x, min_y = np.min(kp2ds_face, axis=0)
    max_x, max_y = np.max(kp2ds_face, axis=0)

    initial_width = max_x - min_x
    initial_height = max_y - min_y

    initial_area = initial_width * initial_height

    expanded_area = initial_area * scale

    new_width = np.sqrt(expanded_area * (initial_width / initial_height))
    new_height = np.sqrt(expanded_area * (initial_height / initial_width))

    delta_width = (new_width - initial_width) / 2
    delta_height = (new_height - initial_height) / 4

    if ratio_aug:
        if random.random() > 0.5:
            delta_width += random.uniform(0, initial_width // 10)
        else:
            delta_height += random.uniform(0, initial_height // 10)

    expanded_min_x = max(min_x - delta_width, 0)
    expanded_max_x = min(max_x + delta_width, w)
    expanded_min_y = max(min_y - 3 * delta_height, 0)
    expanded_max_y = min(max_y + delta_height, h)

    return [
        int(expanded_min_x),
        int(expanded_max_x),
        int(expanded_min_y),
        int(expanded_max_y),
    ]


def calculate_new_size(orig_w, orig_h, target_area, divisor=64):

    target_ratio = orig_w / orig_h

    def check_valid(w, h):

        if w <= 0 or h <= 0:
            return False
        return w * h <= target_area and w % divisor == 0 and h % divisor == 0

    def get_ratio_diff(w, h):

        return abs(w / h - target_ratio)

    def round_to_64(value, round_up=False, divisor=64):

        if round_up:
            return divisor * ((value + (divisor - 1)) // divisor)
        return divisor * (value // divisor)

    possible_sizes = []

    max_area_h = int(np.sqrt(target_area / target_ratio))
    max_area_w = int(max_area_h * target_ratio)

    max_h = round_to_64(max_area_h, round_up=True, divisor=divisor)
    max_w = round_to_64(max_area_w, round_up=True, divisor=divisor)

    for h in range(divisor, max_h + divisor, divisor):
        ideal_w = h * target_ratio

        w_down = round_to_64(ideal_w)
        w_up = round_to_64(ideal_w, round_up=True)

        for w in [w_down, w_up]:
            if check_valid(w, h, divisor):
                possible_sizes.append((w, h, get_ratio_diff(w, h)))

    if not possible_sizes:
        raise ValueError("Can not find suitable size")

    possible_sizes.sort(key=lambda x: (-x[0] * x[1], x[2]))

    best_w, best_h, _ = possible_sizes[0]
    return int(best_w), int(best_h)


def resize_by_area(
    image, target_area, keep_aspect_ratio=True, divisor=64, padding_color=(0, 0, 0)
):
    h, w = image.shape[:2]
    try:
        new_w, new_h = calculate_new_size(w, h, target_area, divisor)
    except:
        aspect_ratio = w / h

        if keep_aspect_ratio:
            new_h = math.sqrt(target_area / aspect_ratio)
            new_w = target_area / new_h
        else:
            new_w = new_h = math.sqrt(target_area)

        new_w, new_h = int((new_w // divisor) * divisor), int(
            (new_h // divisor) * divisor
        )

    interpolation = cv2.INTER_AREA if (new_w * new_h < w * h) else cv2.INTER_LINEAR

    resized_image = padding_resize(
        image,
        height=new_h,
        width=new_w,
        padding_color=padding_color,
        interpolation=interpolation,
    )
    return resized_image


def padding_resize(
    img_ori,
    height=512,
    width=512,
    padding_color=(0, 0, 0),
    interpolation=cv2.INTER_LINEAR,
):
    ori_height = img_ori.shape[0]
    ori_width = img_ori.shape[1]
    channel = img_ori.shape[2]

    img_pad = np.zeros((height, width, channel))
    if channel == 1:
        img_pad[:, :, 0] = padding_color[0]
    else:
        img_pad[:, :, 0] = padding_color[0]
        img_pad[:, :, 1] = padding_color[1]
        img_pad[:, :, 2] = padding_color[2]

    if (ori_height / ori_width) > (height / width):
        new_width = int(height / ori_height * ori_width)
        img = cv2.resize(img_ori, (new_width, height), interpolation=interpolation)
        padding = int((width - new_width) / 2)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        img_pad[:, padding : padding + new_width, :] = img
    else:
        new_height = int(width / ori_width * ori_height)
        img = cv2.resize(img_ori, (width, new_height), interpolation=interpolation)
        padding = int((height - new_height) / 2)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        img_pad[padding : padding + new_height, :, :] = img

    img_pad = np.uint8(img_pad)

    return img_pad


def get_frame_indices(frame_num, video_fps, clip_length, train_fps):

    start_frame = 0
    times = np.arange(0, clip_length) / train_fps
    frame_indices = start_frame + np.round(times * video_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, frame_num - 1)

    return frame_indices.tolist()


def get_face_bboxes(kp2ds, scale, image_shape):
    h, w = image_shape
    kp2ds_face = kp2ds.copy()[1:] * (w, h)

    min_x, min_y = np.min(kp2ds_face, axis=0)
    max_x, max_y = np.max(kp2ds_face, axis=0)

    initial_width = max_x - min_x
    initial_height = max_y - min_y

    initial_area = initial_width * initial_height

    expanded_area = initial_area * scale

    new_width = np.sqrt(expanded_area * (initial_width / initial_height))
    new_height = np.sqrt(expanded_area * (initial_height / initial_width))

    delta_width = (new_width - initial_width) / 2
    delta_height = (new_height - initial_height) / 4

    expanded_min_x = max(min_x - delta_width, 0)
    expanded_max_x = min(max_x + delta_width, w)
    expanded_min_y = max(min_y - 3 * delta_height, 0)
    expanded_max_y = min(max_y + delta_height, h)

    return [
        int(expanded_min_x),
        int(expanded_max_x),
        int(expanded_min_y),
        int(expanded_max_y),
    ]


def box_convert_simple(box, convert_type="xyxy2xywh"):
    if convert_type == "xyxy2xywh":
        return [box[0], box[1], box[2] - box[0], box[3] - box[1]]
    elif convert_type == "xywh2xyxy":
        return [box[0], box[1], box[2] + box[0], box[3] + box[1]]
    elif convert_type == "xyxy2ctwh":
        return [
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2,
            box[2] - box[0],
            box[3] - box[1],
        ]
    elif convert_type == "ctwh2xyxy":
        return [
            box[0] - box[2] // 2,
            box[1] - box[3] // 2,
            box[0] + (box[2] - box[2] // 2),
            box[1] + (box[3] - box[3] // 2),
        ]


def read_img(image, convert="RGB", check_exist=False):
    if isinstance(image, str):
        if check_exist and not os.exists(image):
            return None
        try:
            img = Image.open(image)
            if convert:
                img = img.convert(convert)
        except:
            raise IOError("File error: ", image)
        return np.asarray(img)
    else:
        if isinstance(image, np.ndarray):
            if convert:
                return image[..., ::-1]
        else:
            if convert:
                img = img.convert(convert)
            return np.asarray(img)


class AAPoseMeta:
    def __init__(self, meta=None, kp2ds=None):
        self.image_id = ""
        self.height = 0
        self.width = 0

        self.kps_body: np.ndarray = None
        self.kps_lhand: np.ndarray = None
        self.kps_rhand: np.ndarray = None
        self.kps_face: np.ndarray = None
        self.kps_body_p: np.ndarray = None
        self.kps_lhand_p: np.ndarray = None
        self.kps_rhand_p: np.ndarray = None
        self.kps_face_p: np.ndarray = None

        if meta is not None:
            self.load_from_meta(meta)
        elif kp2ds is not None:
            self.load_from_kp2ds(kp2ds)

    def is_valid(self, kp, p, threshold):
        x, y = kp
        if x < 0 or y < 0 or x > self.width or y > self.height or p < threshold:
            return False
        else:
            return True

    def get_bbox(self, kp, kp_p, threshold=0.5):
        kps = kp[kp_p > threshold]
        if kps.size == 0:
            return 0, 0, 0, 0
        x0, y0 = kps.min(axis=0)
        x1, y1 = kps.max(axis=0)
        return x0, y0, x1, y1

    def crop(self, x0, y0, x1, y1):
        all_kps = [self.kps_body, self.kps_lhand, self.kps_rhand, self.kps_face]
        for kps in all_kps:
            if kps is not None:
                kps[:, 0] -= x0
                kps[:, 1] -= y0
        self.width = x1 - x0
        self.height = y1 - y0
        return self

    def resize(self, width, height):
        scale_x = width / self.width
        scale_y = height / self.height
        all_kps = [self.kps_body, self.kps_lhand, self.kps_rhand, self.kps_face]
        for kps in all_kps:
            if kps is not None:
                kps[:, 0] *= scale_x
                kps[:, 1] *= scale_y
        self.width = width
        self.height = height
        return self

    def get_kps_body_with_p(self, normalize=False):
        kps_body = self.kps_body.copy()
        if normalize:
            kps_body = kps_body / np.array([self.width, self.height])

        return np.concatenate([kps_body, self.kps_body_p[:, None]])

    @staticmethod
    def from_kps_face(kps_face: np.ndarray, height: int, width: int):

        pose_meta = AAPoseMeta()
        pose_meta.kps_face = kps_face[:, :2]
        if kps_face.shape[1] == 3:
            pose_meta.kps_face_p = kps_face[:, 2]
        else:
            pose_meta.kps_face_p = kps_face[:, 0] * 0 + 1
        pose_meta.height = height
        pose_meta.width = width
        return pose_meta

    @staticmethod
    def from_kps_body(kps_body: np.ndarray, height: int, width: int):

        pose_meta = AAPoseMeta()
        pose_meta.kps_body = kps_body[:, :2]
        pose_meta.kps_body_p = kps_body[:, 2]
        pose_meta.height = height
        pose_meta.width = width
        return pose_meta

    @staticmethod
    def from_humanapi_meta(meta):
        pose_meta = AAPoseMeta()
        width, height = meta["width"], meta["height"]
        pose_meta.width = width
        pose_meta.height = height
        pose_meta.kps_body = meta["keypoints_body"][:, :2] * (width, height)
        pose_meta.kps_body_p = meta["keypoints_body"][:, 2]
        pose_meta.kps_lhand = meta["keypoints_left_hand"][:, :2] * (width, height)
        pose_meta.kps_lhand_p = meta["keypoints_left_hand"][:, 2]
        pose_meta.kps_rhand = meta["keypoints_right_hand"][:, :2] * (width, height)
        pose_meta.kps_rhand_p = meta["keypoints_right_hand"][:, 2]
        if "keypoints_face" in meta:
            pose_meta.kps_face = meta["keypoints_face"][:, :2] * (width, height)
            pose_meta.kps_face_p = meta["keypoints_face"][:, 2]
        return pose_meta

    def load_from_meta(self, meta, norm_body=True, norm_hand=False):

        self.image_id = meta.get("image_id", "00000.png")
        self.height = meta["height"]
        self.width = meta["width"]
        kps_body_p = []
        kps_body = []
        for kp in meta["keypoints_body"]:
            if kp is None:
                kps_body.append([0, 0])
                kps_body_p.append(0)
            else:
                kps_body.append(kp)
                kps_body_p.append(1)

        self.kps_body = np.array(kps_body)
        self.kps_body[:, 0] *= self.width
        self.kps_body[:, 1] *= self.height
        self.kps_body_p = np.array(kps_body_p)

        self.kps_lhand = np.array(meta["keypoints_left_hand"])[:, :2]
        self.kps_lhand_p = np.array(meta["keypoints_left_hand"])[:, 2]
        self.kps_rhand = np.array(meta["keypoints_right_hand"])[:, :2]
        self.kps_rhand_p = np.array(meta["keypoints_right_hand"])[:, 2]

    @staticmethod
    def load_from_kp2ds(kp2ds: List[np.ndarray], width: int, height: int):
        """input 133x3 numpy keypoints and output AAPoseMeta

        Args:
            kp2ds (List[np.ndarray]): _description_
            width (int): _description_
            height (int): _description_

        Returns:
            _type_: _description_
        """
        pose_meta = AAPoseMeta()
        pose_meta.width = width
        pose_meta.height = height
        kps_body = (
            kp2ds[[0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 20]]
            + kp2ds[
                [0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 18, 21]
            ]
        ) / 2
        kps_lhand = kp2ds[91:112]
        kps_rhand = kp2ds[112:133]
        kps_face = np.concatenate([kp2ds[23 : 23 + 68], kp2ds[1:3]], axis=0)
        pose_meta.kps_body = kps_body[:, :2]
        pose_meta.kps_body_p = kps_body[:, 2]
        pose_meta.kps_lhand = kps_lhand[:, :2]
        pose_meta.kps_lhand_p = kps_lhand[:, 2]
        pose_meta.kps_rhand = kps_rhand[:, :2]
        pose_meta.kps_rhand_p = kps_rhand[:, 2]
        pose_meta.kps_face = kps_face[:, :2]
        pose_meta.kps_face_p = kps_face[:, 2]
        return pose_meta

    @staticmethod
    def from_dwpose(dwpose_det_res, height, width):
        pose_meta = AAPoseMeta()
        pose_meta.kps_body = dwpose_det_res["bodies"]["candidate"]
        pose_meta.kps_body_p = dwpose_det_res["bodies"]["score"]
        pose_meta.kps_body[:, 0] *= width
        pose_meta.kps_body[:, 1] *= height

        pose_meta.kps_lhand, pose_meta.kps_rhand = dwpose_det_res["hands"]
        pose_meta.kps_lhand[:, 0] *= width
        pose_meta.kps_lhand[:, 1] *= height
        pose_meta.kps_rhand[:, 0] *= width
        pose_meta.kps_rhand[:, 1] *= height
        pose_meta.kps_lhand_p, pose_meta.kps_rhand_p = dwpose_det_res["hands_score"]

        pose_meta.kps_face = dwpose_det_res["faces"][0]
        pose_meta.kps_face[:, 0] *= width
        pose_meta.kps_face[:, 1] *= height
        pose_meta.kps_face_p = dwpose_det_res["faces_score"][0]
        return pose_meta

    def save_json(self):
        pass

    def draw_aapose(
        self, img, threshold=0.5, stick_width_norm=200, draw_hand=True, draw_head=True
    ):
        from .human_visualization import draw_aapose_by_meta

        return draw_aapose_by_meta(
            img, self, threshold, stick_width_norm, draw_hand, draw_head
        )

    def translate(self, x0, y0):
        all_kps = [self.kps_body, self.kps_lhand, self.kps_rhand, self.kps_face]
        for kps in all_kps:
            if kps is not None:
                kps[:, 0] -= x0
                kps[:, 1] -= y0

    def scale(self, sx, sy):
        all_kps = [self.kps_body, self.kps_lhand, self.kps_rhand, self.kps_face]
        for kps in all_kps:
            if kps is not None:
                kps[:, 0] *= sx
                kps[:, 1] *= sy

    def padding_resize2(self, height=512, width=512):
        """kps will be changed inplace"""

        all_kps = [self.kps_body, self.kps_lhand, self.kps_rhand, self.kps_face]

        ori_height, ori_width = self.height, self.width

        if (ori_height / ori_width) > (height / width):
            new_width = int(height / ori_height * ori_width)
            padding = int((width - new_width) / 2)
            padding_width = padding
            padding_height = 0
            scale = height / ori_height

            for kps in all_kps:
                if kps is not None:
                    kps[:, 0] = kps[:, 0] * scale + padding
                    kps[:, 1] = kps[:, 1] * scale

        else:
            new_height = int(width / ori_width * ori_height)
            padding = int((height - new_height) / 2)
            padding_width = 0
            padding_height = padding
            scale = width / ori_width
            for kps in all_kps:
                if kps is not None:
                    kps[:, 1] = kps[:, 1] * scale + padding
                    kps[:, 0] = kps[:, 0] * scale

        self.width = width
        self.height = height
        return self


def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    # scale = scale * 200.0

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords


def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps, np.ndarray), "heatmaps should be numpy.ndarray"
    assert heatmaps.ndim == 4, "batch_images should be 4-ndim"

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def _taylor(heatmap, coord):
    """Distribution aware coordinate decoding method.

    Note:
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

    Returns:
        np.ndarray[2,]: Updated coordinates.
    """
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
            heatmap[py + 1][px + 1]
            - heatmap[py - 1][px + 1]
            - heatmap[py + 1][px - 1]
            + heatmap[py - 1][px - 1]
        )
        dyy = 0.25 * (
            heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] + heatmap[py - 2 * 1][px]
        )
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        - batch size: B
        - num keypoints: K
        - num persons: N
        - height of heatmaps: H
        - width of heatmaps: W

        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        np.ndarray([N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert B == 1 or B == N
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)

    batch_heatmaps_pad = np.pad(
        batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge"
    ).flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum("ijmn,ijnk->ijmk", hessian, derivative).squeeze()
    return coords


def _gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([N, K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border), dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


def keypoints_from_heatmaps(
    heatmaps,
    center,
    scale,
    unbiased=False,
    post_process="default",
    kernel=11,
    valid_radius_factor=0.0546875,
    use_udp=False,
    target_type="GaussianHeatmap",
):
    """Get final keypoint predictions from heatmaps and transform them back to
    the image.

    Note:
        - batch size: N
        - num keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        post_process (str/None): Choice of methods to post-process
            heatmaps. Currently supported: None, 'default', 'unbiased',
            'megvii'.
        unbiased (bool): Option to use unbiased decoding. Mutually
            exclusive with megvii.
            Note: this arg is deprecated and unbiased=True can be replaced
            by post_process='unbiased'
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.
        valid_radius_factor (float): The radius factor of the positive area
            in classification heatmap for UDP.
        use_udp (bool): Use unbiased data processing.
        target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
            GaussianHeatmap: Classification target with gaussian distribution.
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        tuple: A tuple containing keypoint predictions and scores.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    # Avoid being affected
    heatmaps = heatmaps.copy()

    # detect conflicts
    if unbiased:
        assert post_process not in [False, None, "megvii"]
    if post_process in ["megvii", "unbiased"]:
        assert kernel > 0
    if use_udp:
        assert not post_process == "megvii"

    # normalize configs
    if post_process is False:
        warnings.warn(
            "post_process=False is deprecated, " "please use post_process=None instead",
            DeprecationWarning,
        )
        post_process = None
    elif post_process is True:
        if unbiased is True:
            warnings.warn(
                "post_process=True, unbiased=True is deprecated,"
                " please use post_process='unbiased' instead",
                DeprecationWarning,
            )
            post_process = "unbiased"
        else:
            warnings.warn(
                "post_process=True, unbiased=False is deprecated, "
                "please use post_process='default' instead",
                DeprecationWarning,
            )
            post_process = "default"
    elif post_process == "default":
        if unbiased is True:
            warnings.warn(
                "unbiased=True is deprecated, please use "
                "post_process='unbiased' instead",
                DeprecationWarning,
            )
            post_process = "unbiased"

    # start processing
    if post_process == "megvii":
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

    N, K, H, W = heatmaps.shape
    if use_udp:
        if target_type.lower() == "GaussianHeatMap".lower():
            preds, maxvals = _get_max_preds(heatmaps)
            preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        elif target_type.lower() == "CombinedTarget".lower():
            for person_heatmaps in heatmaps:
                for i, heatmap in enumerate(person_heatmaps):
                    kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            # valid radius is in direct proportion to the height of heatmap.
            valid_radius = valid_radius_factor * H
            offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            heatmaps = heatmaps[:, ::3, :]
            preds, maxvals = _get_max_preds(heatmaps)
            index = preds[..., 0] + preds[..., 1] * W
            index += W * H * np.arange(0, N * K / 3)
            index = index.astype(int).reshape(N, K // 3, 1)
            preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        else:
            raise ValueError(
                "target_type should be either " "'GaussianHeatmap' or 'CombinedTarget'"
            )
    else:
        preds, maxvals = _get_max_preds(heatmaps)
        if post_process == "unbiased":  # alleviate biased coordinate
            # apply Gaussian distribution modulation.
            heatmaps = np.log(np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array(
                            [
                                heatmap[py][px + 1] - heatmap[py][px - 1],
                                heatmap[py + 1][px] - heatmap[py - 1][px],
                            ]
                        )
                        preds[n][k] += np.sign(diff) * 0.25
                        if post_process == "megvii":
                            preds[n][k] += 0.5

    # Transform back to the image
    for i in range(N):
        preds[i] = transform_preds(
            preds[i], center[i], scale[i], [W, H], use_udp=use_udp
        )

    if post_process == "megvii":
        maxvals = maxvals / 255.0 + 0.5

    return preds, maxvals


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    # res: (height, width), (rows, cols)
    crop_aspect_ratio = res[0] / float(res[1])
    h = 200 * scale
    w = h / crop_aspect_ratio
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + 0.5)
    t[1, 2] = res[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.0]).T
    new_pt = np.dot(t, new_pt)
    return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1


def bbox_from_detector(bbox, input_resolution=(224, 224), rescale=1.25):
    """
    Get center and scale of bounding box from bounding box.
    The expected format is [min_x, min_y, max_x, max_y].
    """
    CROP_IMG_HEIGHT, CROP_IMG_WIDTH = input_resolution
    CROP_ASPECT_RATIO = CROP_IMG_HEIGHT / float(CROP_IMG_WIDTH)

    # center
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    center = np.array([center_x, center_y])

    # scale
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    bbox_size = max(bbox_w * CROP_ASPECT_RATIO, bbox_h)

    scale = np.array([bbox_size / CROP_ASPECT_RATIO, bbox_size]) / 200.0
    # scale = bbox_size / 200.0
    # adjust bounding box tightness
    scale *= rescale
    return center, scale


def crop(img, center, scale, res):
    """
    Crop image according to the supplied bounding box.
    res: [rows, cols]
    """
    # Upper left point
    ul = np.array(transform([1, 1], center, max(scale), res, invert=1)) - 1
    # Bottom right point
    br = (
        np.array(transform([res[1] + 1, res[0] + 1], center, max(scale), res, invert=1))
        - 1
    )

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.float32)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    try:
        new_img[new_y[0] : new_y[1], new_x[0] : new_x[1]] = img[
            old_y[0] : old_y[1], old_x[0] : old_x[1]
        ]
    except Exception as e:
        print(e)

    new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)
    return new_img, new_shape, (old_x, old_y), (new_x, new_y)  # , ul, br


def split_kp2ds_for_aa(kp2ds, ret_face=False):
    kp2ds_body = (
        kp2ds[[0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 20]]
        + kp2ds[[0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 18, 21]]
    ) / 2
    kp2ds_lhand = kp2ds[91:112]
    kp2ds_rhand = kp2ds[112:133]
    kp2ds_face = kp2ds[22:91]
    if ret_face:
        return (
            kp2ds_body.copy(),
            kp2ds_lhand.copy(),
            kp2ds_rhand.copy(),
            kp2ds_face.copy(),
        )
    return kp2ds_body.copy(), kp2ds_lhand.copy(), kp2ds_rhand.copy()


def load_pose_metas_from_kp2ds_seq(kp2ds_seq, width, height):
    metas = []
    last_kp2ds_body = None
    for kps in kp2ds_seq:
        kps = kps.copy()
        kps[:, 0] /= width
        kps[:, 1] /= height
        kp2ds_body, kp2ds_lhand, kp2ds_rhand, kp2ds_face = split_kp2ds_for_aa(
            kps, ret_face=True
        )

        if kp2ds_body[:, :2].min(axis=1).max() < 0:
            kp2ds_body = last_kp2ds_body
        last_kp2ds_body = kp2ds_body

        meta = {
            "width": width,
            "height": height,
            "keypoints_body": kp2ds_body,
            "keypoints_left_hand": kp2ds_lhand,
            "keypoints_right_hand": kp2ds_rhand,
            "keypoints_face": kp2ds_face,
        }
        metas.append(meta)
    return metas
