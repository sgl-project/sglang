# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import copy
import math
from typing import NamedTuple

import numpy as np
from tqdm import tqdm

from sglang.multimodal_gen.runtime.utils.pose2d import AAPoseMeta

# load skeleton name and bone lines
keypoint_list = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",  # No.4
    "LShoulder",
    "LElbow",
    "LWrist",  # No.7
    "RHip",
    "RKnee",
    "RAnkle",  # No.10
    "LHip",
    "LKnee",
    "LAnkle",  # No.13
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LToe",
    "RToe",
]


limbSeq = [
    [2, 3],
    [2, 6],  # shoulders
    [3, 4],
    [4, 5],  # left arm
    [6, 7],
    [7, 8],  # right arm
    [2, 9],
    [9, 10],
    [10, 11],  # right leg
    [2, 12],
    [12, 13],
    [13, 14],  # left leg
    [2, 1],
    [1, 15],
    [15, 17],
    [1, 16],
    [16, 18],  # face (nose, eyes, ears)
    [14, 19],  # left foot
    [11, 20],  #  right foot
]

eps = 0.01


class Keypoint(NamedTuple):
    x: float
    y: float
    score: float = 1.0
    id: int = -1


# for each limb, calculate src & dst bone's length
# and calculate their ratios
def get_length(skeleton, limb):

    k1_index, k2_index = limb

    H, W = skeleton["height"], skeleton["width"]
    keypoints = skeleton["keypoints_body"]
    keypoint1 = keypoints[k1_index - 1]
    keypoint2 = keypoints[k2_index - 1]

    if keypoint1 is None or keypoint2 is None:
        return None, None, None

    X = np.array([keypoint1[0], keypoint2[0]]) * float(W)
    Y = np.array([keypoint1[1], keypoint2[1]]) * float(H)
    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5

    return X, Y, length


def get_handpose_meta(keypoints, delta, src_H, src_W):

    new_keypoints = []

    for idx, keypoint in enumerate(keypoints):
        if keypoint is None:
            new_keypoints.append(None)
            continue
        if keypoint.score == 0:
            new_keypoints.append(None)
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * src_W + delta[0])
        y = int(y * src_H + delta[1])

        new_keypoints.append(
            Keypoint(
                x=x,
                y=y,
                score=keypoint.score,
            )
        )

    return new_keypoints


def deal_hand_keypoints(hand_res, r_ratio, l_ratio, hand_score_th=0.5):

    left_hand = []
    right_hand = []

    left_delta_x = hand_res["left"][0][0] * (l_ratio - 1)
    left_delta_y = hand_res["left"][0][1] * (l_ratio - 1)

    right_delta_x = hand_res["right"][0][0] * (r_ratio - 1)
    right_delta_y = hand_res["right"][0][1] * (r_ratio - 1)

    length = len(hand_res["left"])

    for i in range(length):
        # left hand
        if hand_res["left"][i][2] < hand_score_th:
            left_hand.append(
                Keypoint(
                    x=-1,
                    y=-1,
                    score=0,
                )
            )
        else:
            left_hand.append(
                Keypoint(
                    x=hand_res["left"][i][0] * l_ratio - left_delta_x,
                    y=hand_res["left"][i][1] * l_ratio - left_delta_y,
                    score=hand_res["left"][i][2],
                )
            )

        # right hand
        if hand_res["right"][i][2] < hand_score_th:
            right_hand.append(
                Keypoint(
                    x=-1,
                    y=-1,
                    score=0,
                )
            )
        else:
            right_hand.append(
                Keypoint(
                    x=hand_res["right"][i][0] * r_ratio - right_delta_x,
                    y=hand_res["right"][i][1] * r_ratio - right_delta_y,
                    score=hand_res["right"][i][2],
                )
            )

    return right_hand, left_hand


def get_scaled_pose(
    canvas,
    src_canvas,
    keypoints,
    keypoints_hand,
    bone_ratio_list,
    delta_ground_x,
    delta_ground_y,
    rescaled_src_ground_x,
    body_flag,
    id,
    scale_min,
    threshold=0.4,
):

    H, W = canvas
    src_H, src_W = src_canvas

    new_length_list = []
    angle_list = []

    # keypoints from 0-1 to H/W range
    for idx in range(len(keypoints)):
        if keypoints[idx] is None or len(keypoints[idx]) == 0:
            continue

        keypoints[idx] = [
            keypoints[idx][0] * src_W,
            keypoints[idx][1] * src_H,
            keypoints[idx][2],
        ]

    # first traverse, get new_length_list and angle_list
    for idx, (k1_index, k2_index) in enumerate(limbSeq):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if (
            keypoint1 is None
            or keypoint2 is None
            or len(keypoint1) == 0
            or len(keypoint2) == 0
        ):
            new_length_list.append(None)
            angle_list.append(None)
            continue

        Y = np.array([keypoint1[0], keypoint2[0]])  # * float(W)
        X = np.array([keypoint1[1], keypoint2[1]])  # * float(H)

        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5

        new_length = length * bone_ratio_list[idx]
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

        new_length_list.append(new_length)
        angle_list.append(angle)

    # Keep foot length within 0.5x calf length
    foot_lower_leg_ratio = 0.5
    if new_length_list[8] != None and new_length_list[18] != None:
        if new_length_list[18] > new_length_list[8] * foot_lower_leg_ratio:
            new_length_list[18] = new_length_list[8] * foot_lower_leg_ratio

    if new_length_list[11] != None and new_length_list[17] != None:
        if new_length_list[17] > new_length_list[11] * foot_lower_leg_ratio:
            new_length_list[17] = new_length_list[11] * foot_lower_leg_ratio

    # second traverse, calculate new keypoints
    rescale_keypoints = keypoints.copy()

    for idx, (k1_index, k2_index) in enumerate(limbSeq):
        # update dst_keypoints
        start_keypoint = rescale_keypoints[k1_index - 1]
        new_length = new_length_list[idx]
        angle = angle_list[idx]

        if (
            rescale_keypoints[k1_index - 1] is None
            or rescale_keypoints[k2_index - 1] is None
            or len(rescale_keypoints[k1_index - 1]) == 0
            or len(rescale_keypoints[k2_index - 1]) == 0
        ):
            continue

        # calculate end_keypoint
        delta_x = new_length * math.cos(math.radians(angle))
        delta_y = new_length * math.sin(math.radians(angle))

        end_keypoint_x = start_keypoint[0] - delta_x
        end_keypoint_y = start_keypoint[1] - delta_y

        # update keypoints
        rescale_keypoints[k2_index - 1] = [
            end_keypoint_x,
            end_keypoint_y,
            rescale_keypoints[k2_index - 1][2],
        ]

    if id == 0:
        if (
            body_flag == "full_body"
            and rescale_keypoints[8] != None
            and rescale_keypoints[11] != None
        ):
            delta_ground_x_offset_first_frame = (
                rescale_keypoints[8][0] + rescale_keypoints[11][0]
            ) / 2 - rescaled_src_ground_x
            delta_ground_x += delta_ground_x_offset_first_frame
        elif body_flag == "half_body" and rescale_keypoints[1] != None:
            delta_ground_x_offset_first_frame = (
                rescale_keypoints[1][0] - rescaled_src_ground_x
            )
            delta_ground_x += delta_ground_x_offset_first_frame

    # offset all keypoints
    for idx in range(len(rescale_keypoints)):
        if rescale_keypoints[idx] is None or len(rescale_keypoints[idx]) == 0:
            continue
        rescale_keypoints[idx][0] -= delta_ground_x
        rescale_keypoints[idx][1] -= delta_ground_y

        # rescale keypoints to original size
        rescale_keypoints[idx][0] /= scale_min
        rescale_keypoints[idx][1] /= scale_min

    # Scale hand proportions based on body skeletal ratios
    r_ratio = max(bone_ratio_list[0], bone_ratio_list[1]) / scale_min
    l_ratio = max(bone_ratio_list[0], bone_ratio_list[1]) / scale_min
    left_hand, right_hand = deal_hand_keypoints(
        keypoints_hand, r_ratio, l_ratio, hand_score_th=threshold
    )

    left_hand_new = left_hand.copy()
    right_hand_new = right_hand.copy()

    if rescale_keypoints[4] == None and rescale_keypoints[7] == None:
        pass

    elif rescale_keypoints[4] == None and rescale_keypoints[7] != None:
        right_hand_delta = np.array(rescale_keypoints[7][:2]) - np.array(
            keypoints[7][:2]
        )
        right_hand_new = get_handpose_meta(right_hand, right_hand_delta, src_H, src_W)

    elif rescale_keypoints[4] != None and rescale_keypoints[7] == None:
        left_hand_delta = np.array(rescale_keypoints[4][:2]) - np.array(
            keypoints[4][:2]
        )
        left_hand_new = get_handpose_meta(left_hand, left_hand_delta, src_H, src_W)

    else:
        # get left_hand and right_hand offset
        left_hand_delta = np.array(rescale_keypoints[4][:2]) - np.array(
            keypoints[4][:2]
        )
        right_hand_delta = np.array(rescale_keypoints[7][:2]) - np.array(
            keypoints[7][:2]
        )

        if keypoints[4][0] != None and left_hand[0].x != -1:
            left_hand_root_offset = np.array(
                (
                    keypoints[4][0] - left_hand[0].x * src_W,
                    keypoints[4][1] - left_hand[0].y * src_H,
                )
            )
            left_hand_delta += left_hand_root_offset

        if keypoints[7][0] != None and right_hand[0].x != -1:
            right_hand_root_offset = np.array(
                (
                    keypoints[7][0] - right_hand[0].x * src_W,
                    keypoints[7][1] - right_hand[0].y * src_H,
                )
            )
            right_hand_delta += right_hand_root_offset

        dis_left_hand = (
            (keypoints[4][0] - left_hand[0].x * src_W) ** 2
            + (keypoints[4][1] - left_hand[0].y * src_H) ** 2
        ) ** 0.5
        dis_right_hand = (
            (keypoints[7][0] - left_hand[0].x * src_W) ** 2
            + (keypoints[7][1] - left_hand[0].y * src_H) ** 2
        ) ** 0.5

        if dis_left_hand > dis_right_hand:
            right_hand_new = get_handpose_meta(
                left_hand, right_hand_delta, src_H, src_W
            )
            left_hand_new = get_handpose_meta(right_hand, left_hand_delta, src_H, src_W)
        else:
            left_hand_new = get_handpose_meta(left_hand, left_hand_delta, src_H, src_W)
            right_hand_new = get_handpose_meta(
                right_hand, right_hand_delta, src_H, src_W
            )

    # get normalized keypoints_body
    norm_body_keypoints = []
    for body_keypoint in rescale_keypoints:
        if body_keypoint != None:
            norm_body_keypoints.append(
                [body_keypoint[0] / W, body_keypoint[1] / H, body_keypoint[2]]
            )
        else:
            norm_body_keypoints.append(None)

    frame_info = {
        "height": H,
        "width": W,
        "keypoints_body": norm_body_keypoints,
        "keypoints_left_hand": left_hand_new,
        "keypoints_right_hand": right_hand_new,
    }

    return frame_info


def rescale_skeleton(H, W, keypoints, bone_ratio_list):

    rescale_keypoints = keypoints.copy()

    new_length_list = []
    angle_list = []

    # keypoints from 0-1 to H/W range
    for idx in range(len(rescale_keypoints)):
        if rescale_keypoints[idx] is None or len(rescale_keypoints[idx]) == 0:
            continue

        rescale_keypoints[idx] = [
            rescale_keypoints[idx][0] * W,
            rescale_keypoints[idx][1] * H,
        ]

    # first traverse, get new_length_list and angle_list
    for idx, (k1_index, k2_index) in enumerate(limbSeq):
        keypoint1 = rescale_keypoints[k1_index - 1]
        keypoint2 = rescale_keypoints[k2_index - 1]

        if (
            keypoint1 is None
            or keypoint2 is None
            or len(keypoint1) == 0
            or len(keypoint2) == 0
        ):
            new_length_list.append(None)
            angle_list.append(None)
            continue

        Y = np.array([keypoint1[0], keypoint2[0]])  # * float(W)
        X = np.array([keypoint1[1], keypoint2[1]])  # * float(H)

        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5

        new_length = length * bone_ratio_list[idx]
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

        new_length_list.append(new_length)
        angle_list.append(angle)

    # # second traverse, calculate new keypoints
    for idx, (k1_index, k2_index) in enumerate(limbSeq):
        # update dst_keypoints
        start_keypoint = rescale_keypoints[k1_index - 1]
        new_length = new_length_list[idx]
        angle = angle_list[idx]

        if (
            rescale_keypoints[k1_index - 1] is None
            or rescale_keypoints[k2_index - 1] is None
            or len(rescale_keypoints[k1_index - 1]) == 0
            or len(rescale_keypoints[k2_index - 1]) == 0
        ):
            continue

        # calculate end_keypoint
        delta_x = new_length * math.cos(math.radians(angle))
        delta_y = new_length * math.sin(math.radians(angle))

        end_keypoint_x = start_keypoint[0] - delta_x
        end_keypoint_y = start_keypoint[1] - delta_y

        # update keypoints
        rescale_keypoints[k2_index - 1] = [end_keypoint_x, end_keypoint_y]

    return rescale_keypoints


def fix_lack_keypoints_use_sym(skeleton):

    keypoints = skeleton["keypoints_body"]
    H, W = skeleton["height"], skeleton["width"]

    limb_points_list = [
        [3, 4, 5],
        [6, 7, 8],
        [12, 13, 14, 19],
        [9, 10, 11, 20],
    ]

    for limb_points in limb_points_list:
        miss_flag = False
        for point in limb_points:
            if keypoints[point - 1] is None:
                miss_flag = True
                continue
            if miss_flag:
                skeleton["keypoints_body"][point - 1] = None

    repair_limb_seq_left = [
        [3, 4],
        [4, 5],  # left arm
        [12, 13],
        [13, 14],  # left leg
        [14, 19],  # left foot
    ]

    repair_limb_seq_right = [
        [6, 7],
        [7, 8],  # right arm
        [9, 10],
        [10, 11],  # right leg
        [11, 20],  # right foot
    ]

    repair_limb_seq = [repair_limb_seq_left, repair_limb_seq_right]

    for idx_part, part in enumerate(repair_limb_seq):
        for idx, limb in enumerate(part):

            k1_index, k2_index = limb
            keypoint1 = keypoints[k1_index - 1]
            keypoint2 = keypoints[k2_index - 1]

            if keypoint1 != None and keypoint2 is None:
                # reference to symmetric limb
                sym_limb = repair_limb_seq[1 - idx_part][idx]
                k1_index_sym, k2_index_sym = sym_limb
                keypoint1_sym = keypoints[k1_index_sym - 1]
                keypoint2_sym = keypoints[k2_index_sym - 1]
                ref_length = 0

                if keypoint1_sym != None and keypoint2_sym != None:
                    X = np.array([keypoint1_sym[0], keypoint2_sym[0]]) * float(W)
                    Y = np.array([keypoint1_sym[1], keypoint2_sym[1]]) * float(H)
                    ref_length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                else:
                    ref_length_left, ref_length_right = 0, 0
                    if keypoints[1] != None and keypoints[8] != None:
                        X = np.array([keypoints[1][0], keypoints[8][0]]) * float(W)
                        Y = np.array([keypoints[1][1], keypoints[8][1]]) * float(H)
                        ref_length_left = (
                            (X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2
                        ) ** 0.5
                        if idx <= 1:  # arms
                            ref_length_left /= 2

                    if keypoints[1] != None and keypoints[11] != None:
                        X = np.array([keypoints[1][0], keypoints[11][0]]) * float(W)
                        Y = np.array([keypoints[1][1], keypoints[11][1]]) * float(H)
                        ref_length_right = (
                            (X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2
                        ) ** 0.5
                        if idx <= 1:  # arms
                            ref_length_right /= 2
                        elif idx == 4:  # foot
                            ref_length_right /= 5

                    ref_length = max(ref_length_left, ref_length_right)

                if ref_length != 0:
                    skeleton["keypoints_body"][k2_index - 1] = [0, 0]  # init
                    skeleton["keypoints_body"][k2_index - 1][0] = skeleton[
                        "keypoints_body"
                    ][k1_index - 1][0]
                    skeleton["keypoints_body"][k2_index - 1][1] = (
                        skeleton["keypoints_body"][k1_index - 1][1] + ref_length / H
                    )
    return skeleton


def rescale_shorten_skeleton(ratio_list, src_length_list, dst_length_list):

    modify_bone_list = [[0, 1], [2, 4], [3, 5], [6, 9], [7, 10], [8, 11], [17, 18]]

    for modify_bone in modify_bone_list:
        new_ratio = max(ratio_list[modify_bone[0]], ratio_list[modify_bone[1]])
        ratio_list[modify_bone[0]] = new_ratio
        ratio_list[modify_bone[1]] = new_ratio

    if ratio_list[13] != None and ratio_list[15] != None:
        ratio_eye_avg = (ratio_list[13] + ratio_list[15]) / 2
        ratio_list[13] = ratio_eye_avg
        ratio_list[15] = ratio_eye_avg

    if ratio_list[14] != None and ratio_list[16] != None:
        ratio_eye_avg = (ratio_list[14] + ratio_list[16]) / 2
        ratio_list[14] = ratio_eye_avg
        ratio_list[16] = ratio_eye_avg

    return ratio_list, src_length_list, dst_length_list


def check_full_body(keypoints, threshold=0.4):

    body_flag = "half_body"

    # 1. If ankle points exist, confidence is greater than the threshold, and points do not exceed the frame, return full_body
    if (
        keypoints[10] != None
        and keypoints[13] != None
        and keypoints[8] != None
        and keypoints[11] != None
    ):
        if (
            (keypoints[10][1] <= 1 and keypoints[13][1] <= 1)
            and (keypoints[10][2] >= threshold and keypoints[13][2] >= threshold)
            and (keypoints[8][1] <= 1 and keypoints[11][1] <= 1)
            and (keypoints[8][2] >= threshold and keypoints[11][2] >= threshold)
        ):
            body_flag = "full_body"
            return body_flag

    # 2. If hip points exist, return three_quarter_body
    if keypoints[8] != None and keypoints[11] != None:
        if (keypoints[8][1] <= 1 and keypoints[11][1] <= 1) and (
            keypoints[8][2] >= threshold and keypoints[11][2] >= threshold
        ):
            body_flag = "three_quarter_body"
            return body_flag

    return body_flag


def check_full_body_both(flag1, flag2):
    body_flag_dict = {"full_body": 2, "three_quarter_body": 1, "half_body": 0}

    body_flag_dict_reverse = {2: "full_body", 1: "three_quarter_body", 0: "half_body"}

    flag1_num = body_flag_dict[flag1]
    flag2_num = body_flag_dict[flag2]
    flag_both_num = min(flag1_num, flag2_num)
    return body_flag_dict_reverse[flag_both_num]


def write_to_poses(
    data_to_json,
    none_idx,
    dst_shape,
    bone_ratio_list,
    delta_ground_x,
    delta_ground_y,
    rescaled_src_ground_x,
    body_flag,
    scale_min,
):
    outputs = []
    length = len(data_to_json)
    for id in tqdm(range(length)):

        src_height, src_width = data_to_json[id]["height"], data_to_json[id]["width"]
        width, height = dst_shape
        keypoints = data_to_json[id]["keypoints_body"]
        for idx in range(len(keypoints)):
            if idx in none_idx:
                keypoints[idx] = None
        new_keypoints = keypoints.copy()

        # get hand keypoints
        keypoints_hand = {
            "left": data_to_json[id]["keypoints_left_hand"],
            "right": data_to_json[id]["keypoints_right_hand"],
        }
        # Normalize hand coordinates to 0-1 range
        for hand_idx in range(len(data_to_json[id]["keypoints_left_hand"])):
            data_to_json[id]["keypoints_left_hand"][hand_idx][0] = (
                data_to_json[id]["keypoints_left_hand"][hand_idx][0] / src_width
            )
            data_to_json[id]["keypoints_left_hand"][hand_idx][1] = (
                data_to_json[id]["keypoints_left_hand"][hand_idx][1] / src_height
            )

        for hand_idx in range(len(data_to_json[id]["keypoints_right_hand"])):
            data_to_json[id]["keypoints_right_hand"][hand_idx][0] = (
                data_to_json[id]["keypoints_right_hand"][hand_idx][0] / src_width
            )
            data_to_json[id]["keypoints_right_hand"][hand_idx][1] = (
                data_to_json[id]["keypoints_right_hand"][hand_idx][1] / src_height
            )

        frame_info = get_scaled_pose(
            (height, width),
            (src_height, src_width),
            new_keypoints,
            keypoints_hand,
            bone_ratio_list,
            delta_ground_x,
            delta_ground_y,
            rescaled_src_ground_x,
            body_flag,
            id,
            scale_min,
        )
        outputs.append(frame_info)

    return outputs


def calculate_scale_ratio(skeleton, skeleton_edit, scale_ratio_flag):
    if scale_ratio_flag:

        headw = max(
            skeleton["keypoints_body"][0][0],
            skeleton["keypoints_body"][14][0],
            skeleton["keypoints_body"][15][0],
            skeleton["keypoints_body"][16][0],
            skeleton["keypoints_body"][17][0],
        ) - min(
            skeleton["keypoints_body"][0][0],
            skeleton["keypoints_body"][14][0],
            skeleton["keypoints_body"][15][0],
            skeleton["keypoints_body"][16][0],
            skeleton["keypoints_body"][17][0],
        )
        headw_edit = max(
            skeleton_edit["keypoints_body"][0][0],
            skeleton_edit["keypoints_body"][14][0],
            skeleton_edit["keypoints_body"][15][0],
            skeleton_edit["keypoints_body"][16][0],
            skeleton_edit["keypoints_body"][17][0],
        ) - min(
            skeleton_edit["keypoints_body"][0][0],
            skeleton_edit["keypoints_body"][14][0],
            skeleton_edit["keypoints_body"][15][0],
            skeleton_edit["keypoints_body"][16][0],
            skeleton_edit["keypoints_body"][17][0],
        )
        headw_ratio = headw / headw_edit

        _, _, shoulder = get_length(skeleton, [6, 3])
        _, _, shoulder_edit = get_length(skeleton_edit, [6, 3])
        shoulder_ratio = shoulder / shoulder_edit

        return max(headw_ratio, shoulder_ratio)

    else:
        return 1


def retarget_pose(
    src_skeleton,
    dst_skeleton,
    all_src_skeleton,
    src_skeleton_edit,
    dst_skeleton_edit,
    threshold=0.4,
):

    if src_skeleton_edit is not None and dst_skeleton_edit is not None:
        use_edit_for_base = True
    else:
        use_edit_for_base = False

    src_skeleton_ori = copy.deepcopy(src_skeleton)

    dst_skeleton_ori_h, dst_skeleton_ori_w = (
        dst_skeleton["height"],
        dst_skeleton["width"],
    )
    if (
        src_skeleton["keypoints_body"][0] != None
        and src_skeleton["keypoints_body"][10] != None
        and src_skeleton["keypoints_body"][13] != None
        and dst_skeleton["keypoints_body"][0] != None
        and dst_skeleton["keypoints_body"][10] != None
        and dst_skeleton["keypoints_body"][13] != None
        and src_skeleton["keypoints_body"][0][2] > 0.5
        and src_skeleton["keypoints_body"][10][2] > 0.5
        and src_skeleton["keypoints_body"][13][2] > 0.5
        and dst_skeleton["keypoints_body"][0][2] > 0.5
        and dst_skeleton["keypoints_body"][10][2] > 0.5
        and dst_skeleton["keypoints_body"][13][2] > 0.5
    ):

        src_height = src_skeleton["height"] * abs(
            (
                src_skeleton["keypoints_body"][10][1]
                + src_skeleton["keypoints_body"][13][1]
            )
            / 2
            - src_skeleton["keypoints_body"][0][1]
        )
        dst_height = dst_skeleton["height"] * abs(
            (
                dst_skeleton["keypoints_body"][10][1]
                + dst_skeleton["keypoints_body"][13][1]
            )
            / 2
            - dst_skeleton["keypoints_body"][0][1]
        )
        scale_min = 1.0 * src_height / dst_height
    elif (
        src_skeleton["keypoints_body"][0] != None
        and src_skeleton["keypoints_body"][8] != None
        and src_skeleton["keypoints_body"][11] != None
        and dst_skeleton["keypoints_body"][0] != None
        and dst_skeleton["keypoints_body"][8] != None
        and dst_skeleton["keypoints_body"][11] != None
        and src_skeleton["keypoints_body"][0][2] > 0.5
        and src_skeleton["keypoints_body"][8][2] > 0.5
        and src_skeleton["keypoints_body"][11][2] > 0.5
        and dst_skeleton["keypoints_body"][0][2] > 0.5
        and dst_skeleton["keypoints_body"][8][2] > 0.5
        and dst_skeleton["keypoints_body"][11][2] > 0.5
    ):

        src_height = src_skeleton["height"] * abs(
            (
                src_skeleton["keypoints_body"][8][1]
                + src_skeleton["keypoints_body"][11][1]
            )
            / 2
            - src_skeleton["keypoints_body"][0][1]
        )
        dst_height = dst_skeleton["height"] * abs(
            (
                dst_skeleton["keypoints_body"][8][1]
                + dst_skeleton["keypoints_body"][11][1]
            )
            / 2
            - dst_skeleton["keypoints_body"][0][1]
        )
        scale_min = 1.0 * src_height / dst_height
    else:
        scale_min = np.sqrt(src_skeleton["height"] * src_skeleton["width"]) / np.sqrt(
            dst_skeleton["height"] * dst_skeleton["width"]
        )

    if use_edit_for_base:
        scale_ratio_flag = False
        if (
            src_skeleton_edit["keypoints_body"][0] != None
            and src_skeleton_edit["keypoints_body"][10] != None
            and src_skeleton_edit["keypoints_body"][13] != None
            and dst_skeleton_edit["keypoints_body"][0] != None
            and dst_skeleton_edit["keypoints_body"][10] != None
            and dst_skeleton_edit["keypoints_body"][13] != None
            and src_skeleton_edit["keypoints_body"][0][2] > 0.5
            and src_skeleton_edit["keypoints_body"][10][2] > 0.5
            and src_skeleton_edit["keypoints_body"][13][2] > 0.5
            and dst_skeleton_edit["keypoints_body"][0][2] > 0.5
            and dst_skeleton_edit["keypoints_body"][10][2] > 0.5
            and dst_skeleton_edit["keypoints_body"][13][2] > 0.5
        ):

            src_height_edit = src_skeleton_edit["height"] * abs(
                (
                    src_skeleton_edit["keypoints_body"][10][1]
                    + src_skeleton_edit["keypoints_body"][13][1]
                )
                / 2
                - src_skeleton_edit["keypoints_body"][0][1]
            )
            dst_height_edit = dst_skeleton_edit["height"] * abs(
                (
                    dst_skeleton_edit["keypoints_body"][10][1]
                    + dst_skeleton_edit["keypoints_body"][13][1]
                )
                / 2
                - dst_skeleton_edit["keypoints_body"][0][1]
            )
            scale_min_edit = 1.0 * src_height_edit / dst_height_edit
        elif (
            src_skeleton_edit["keypoints_body"][0] != None
            and src_skeleton_edit["keypoints_body"][8] != None
            and src_skeleton_edit["keypoints_body"][11] != None
            and dst_skeleton_edit["keypoints_body"][0] != None
            and dst_skeleton_edit["keypoints_body"][8] != None
            and dst_skeleton_edit["keypoints_body"][11] != None
            and src_skeleton_edit["keypoints_body"][0][2] > 0.5
            and src_skeleton_edit["keypoints_body"][8][2] > 0.5
            and src_skeleton_edit["keypoints_body"][11][2] > 0.5
            and dst_skeleton_edit["keypoints_body"][0][2] > 0.5
            and dst_skeleton_edit["keypoints_body"][8][2] > 0.5
            and dst_skeleton_edit["keypoints_body"][11][2] > 0.5
        ):

            src_height_edit = src_skeleton_edit["height"] * abs(
                (
                    src_skeleton_edit["keypoints_body"][8][1]
                    + src_skeleton_edit["keypoints_body"][11][1]
                )
                / 2
                - src_skeleton_edit["keypoints_body"][0][1]
            )
            dst_height_edit = dst_skeleton_edit["height"] * abs(
                (
                    dst_skeleton_edit["keypoints_body"][8][1]
                    + dst_skeleton_edit["keypoints_body"][11][1]
                )
                / 2
                - dst_skeleton_edit["keypoints_body"][0][1]
            )
            scale_min_edit = 1.0 * src_height_edit / dst_height_edit
        else:
            scale_min_edit = np.sqrt(
                src_skeleton_edit["height"] * src_skeleton_edit["width"]
            ) / np.sqrt(dst_skeleton_edit["height"] * dst_skeleton_edit["width"])
            scale_ratio_flag = True

        # Flux may change the scale, compensate for it here
        ratio_src = calculate_scale_ratio(
            src_skeleton, src_skeleton_edit, scale_ratio_flag
        )
        ratio_dst = calculate_scale_ratio(
            dst_skeleton, dst_skeleton_edit, scale_ratio_flag
        )

        dst_skeleton_edit["height"] = int(dst_skeleton_edit["height"] * scale_min_edit)
        dst_skeleton_edit["width"] = int(dst_skeleton_edit["width"] * scale_min_edit)
        for idx in range(len(dst_skeleton_edit["keypoints_left_hand"])):
            dst_skeleton_edit["keypoints_left_hand"][idx][0] *= scale_min_edit
            dst_skeleton_edit["keypoints_left_hand"][idx][1] *= scale_min_edit
        for idx in range(len(dst_skeleton_edit["keypoints_right_hand"])):
            dst_skeleton_edit["keypoints_right_hand"][idx][0] *= scale_min_edit
            dst_skeleton_edit["keypoints_right_hand"][idx][1] *= scale_min_edit

    dst_skeleton["height"] = int(dst_skeleton["height"] * scale_min)
    dst_skeleton["width"] = int(dst_skeleton["width"] * scale_min)
    for idx in range(len(dst_skeleton["keypoints_left_hand"])):
        dst_skeleton["keypoints_left_hand"][idx][0] *= scale_min
        dst_skeleton["keypoints_left_hand"][idx][1] *= scale_min
    for idx in range(len(dst_skeleton["keypoints_right_hand"])):
        dst_skeleton["keypoints_right_hand"][idx][0] *= scale_min
        dst_skeleton["keypoints_right_hand"][idx][1] *= scale_min

    dst_body_flag = check_full_body(dst_skeleton["keypoints_body"], threshold)
    src_body_flag = check_full_body(src_skeleton_ori["keypoints_body"], threshold)
    body_flag = check_full_body_both(dst_body_flag, src_body_flag)
    # print('body_flag: ', body_flag)

    if use_edit_for_base:
        src_skeleton_edit = fix_lack_keypoints_use_sym(src_skeleton_edit)
        dst_skeleton_edit = fix_lack_keypoints_use_sym(dst_skeleton_edit)
    else:
        src_skeleton = fix_lack_keypoints_use_sym(src_skeleton)
        dst_skeleton = fix_lack_keypoints_use_sym(dst_skeleton)

    none_idx = []
    for idx in range(len(dst_skeleton["keypoints_body"])):
        if (
            dst_skeleton["keypoints_body"][idx] == None
            or src_skeleton["keypoints_body"][idx] == None
        ):
            src_skeleton["keypoints_body"][idx] = None
            dst_skeleton["keypoints_body"][idx] = None
            none_idx.append(idx)

    # get bone ratio list
    ratio_list, src_length_list, dst_length_list = [], [], []
    for idx, limb in enumerate(limbSeq):
        if use_edit_for_base:
            src_X, src_Y, src_length = get_length(src_skeleton_edit, limb)
            dst_X, dst_Y, dst_length = get_length(dst_skeleton_edit, limb)

            if src_X is None or src_Y is None or dst_X is None or dst_Y is None:
                ratio = -1
            else:
                ratio = 1.0 * dst_length * ratio_dst / src_length / ratio_src

        else:
            src_X, src_Y, src_length = get_length(src_skeleton, limb)
            dst_X, dst_Y, dst_length = get_length(dst_skeleton, limb)

            if src_X is None or src_Y is None or dst_X is None or dst_Y is None:
                ratio = -1
            else:
                ratio = 1.0 * dst_length / src_length

        ratio_list.append(ratio)
        src_length_list.append(src_length)
        dst_length_list.append(dst_length)

    for idx, ratio in enumerate(ratio_list):
        if ratio == -1:
            if ratio_list[0] != -1 and ratio_list[1] != -1:
                ratio_list[idx] = (ratio_list[0] + ratio_list[1]) / 2

    # Consider adding constraints when Flux fails to correct head pose, causing neck issues.
    # if ratio_list[12] > (ratio_list[0]+ratio_list[1])/2*1.25:
    #     ratio_list[12] = (ratio_list[0]+ratio_list[1])/2*1.25

    ratio_list, src_length_list, dst_length_list = rescale_shorten_skeleton(
        ratio_list, src_length_list, dst_length_list
    )

    rescaled_src_skeleton_ori = rescale_skeleton(
        src_skeleton_ori["height"],
        src_skeleton_ori["width"],
        src_skeleton_ori["keypoints_body"],
        ratio_list,
    )

    # get global translation offset_x and offset_y
    if body_flag == "full_body":
        # print('use foot mark.')
        dst_ground_y = (
            max(
                dst_skeleton["keypoints_body"][10][1],
                dst_skeleton["keypoints_body"][13][1],
            )
            * dst_skeleton["height"]
        )
        # The midpoint between toe and ankle
        if (
            dst_skeleton["keypoints_body"][18] != None
            and dst_skeleton["keypoints_body"][19] != None
        ):
            right_foot_mid = (
                dst_skeleton["keypoints_body"][10][1]
                + dst_skeleton["keypoints_body"][19][1]
            ) / 2
            left_foot_mid = (
                dst_skeleton["keypoints_body"][13][1]
                + dst_skeleton["keypoints_body"][18][1]
            ) / 2
            dst_ground_y = max(left_foot_mid, right_foot_mid) * dst_skeleton["height"]

        rescaled_src_ground_y = max(
            rescaled_src_skeleton_ori[10][1], rescaled_src_skeleton_ori[13][1]
        )
        delta_ground_y = rescaled_src_ground_y - dst_ground_y

        dst_ground_x = (
            (
                dst_skeleton["keypoints_body"][8][0]
                + dst_skeleton["keypoints_body"][11][0]
            )
            * dst_skeleton["width"]
            / 2
        )
        rescaled_src_ground_x = (
            rescaled_src_skeleton_ori[8][0] + rescaled_src_skeleton_ori[11][0]
        ) / 2
        delta_ground_x = rescaled_src_ground_x - dst_ground_x
        delta_x, delta_y = delta_ground_x, delta_ground_y

    else:
        # print('use neck mark.')
        # use neck keypoint as mark
        src_neck_y = rescaled_src_skeleton_ori[1][1]
        dst_neck_y = dst_skeleton["keypoints_body"][1][1]
        delta_neck_y = src_neck_y - dst_neck_y * dst_skeleton["height"]

        src_neck_x = rescaled_src_skeleton_ori[1][0]
        dst_neck_x = dst_skeleton["keypoints_body"][1][0]
        delta_neck_x = src_neck_x - dst_neck_x * dst_skeleton["width"]
        delta_x, delta_y = delta_neck_x, delta_neck_y
        rescaled_src_ground_x = src_neck_x

    dst_shape = (dst_skeleton_ori_w, dst_skeleton_ori_h)
    output = write_to_poses(
        all_src_skeleton,
        none_idx,
        dst_shape,
        ratio_list,
        delta_x,
        delta_y,
        rescaled_src_ground_x,
        body_flag,
        scale_min,
    )
    return output


def get_retarget_pose(
    tpl_pose_meta0,
    refer_pose_meta,
    tpl_pose_metas,
    tql_edit_pose_meta0,
    refer_edit_pose_meta,
):

    for key, value in tpl_pose_meta0.items():
        if type(value) is np.ndarray:
            if key in ["keypoints_left_hand", "keypoints_right_hand"]:
                value = value * np.array(
                    [[tpl_pose_meta0["width"], tpl_pose_meta0["height"], 1.0]]
                )
            if not isinstance(value, list):
                value = value.tolist()
        tpl_pose_meta0[key] = value

    for key, value in refer_pose_meta.items():
        if type(value) is np.ndarray:
            if key in ["keypoints_left_hand", "keypoints_right_hand"]:
                value = value * np.array(
                    [[refer_pose_meta["width"], refer_pose_meta["height"], 1.0]]
                )
            if not isinstance(value, list):
                value = value.tolist()
        refer_pose_meta[key] = value

    tpl_pose_metas_new = []
    for meta in tpl_pose_metas:
        for key, value in meta.items():
            if type(value) is np.ndarray:
                if key in ["keypoints_left_hand", "keypoints_right_hand"]:
                    value = value * np.array([[meta["width"], meta["height"], 1.0]])
                if not isinstance(value, list):
                    value = value.tolist()
            meta[key] = value
        tpl_pose_metas_new.append(meta)

    if tql_edit_pose_meta0 is not None:
        for key, value in tql_edit_pose_meta0.items():
            if type(value) is np.ndarray:
                if key in ["keypoints_left_hand", "keypoints_right_hand"]:
                    value = value * np.array(
                        [
                            [
                                tql_edit_pose_meta0["width"],
                                tql_edit_pose_meta0["height"],
                                1.0,
                            ]
                        ]
                    )
                if not isinstance(value, list):
                    value = value.tolist()
            tql_edit_pose_meta0[key] = value

    if refer_edit_pose_meta is not None:
        for key, value in refer_edit_pose_meta.items():
            if type(value) is np.ndarray:
                if key in ["keypoints_left_hand", "keypoints_right_hand"]:
                    value = value * np.array(
                        [
                            [
                                refer_edit_pose_meta["width"],
                                refer_edit_pose_meta["height"],
                                1.0,
                            ]
                        ]
                    )
                if not isinstance(value, list):
                    value = value.tolist()
            refer_edit_pose_meta[key] = value

    retarget_tpl_pose_metas = retarget_pose(
        tpl_pose_meta0,
        refer_pose_meta,
        tpl_pose_metas_new,
        tql_edit_pose_meta0,
        refer_edit_pose_meta,
    )

    pose_metas = []
    for meta in retarget_tpl_pose_metas:
        pose_meta = AAPoseMeta()
        width, height = meta["width"], meta["height"]
        pose_meta.width = width
        pose_meta.height = height
        pose_meta.kps_body = np.array(meta["keypoints_body"])[:, :2] * (width, height)
        pose_meta.kps_body_p = np.array(meta["keypoints_body"])[:, 2]

        kps_lhand = []
        kps_lhand_p = []
        for each_kps_lhand in meta["keypoints_left_hand"]:
            if each_kps_lhand is not None:
                kps_lhand.append([each_kps_lhand.x, each_kps_lhand.y])
                kps_lhand_p.append(each_kps_lhand.score)
            else:
                kps_lhand.append([None, None])
                kps_lhand_p.append(0.0)

        pose_meta.kps_lhand = np.array(kps_lhand)
        pose_meta.kps_lhand_p = np.array(kps_lhand_p)

        kps_rhand = []
        kps_rhand_p = []
        for each_kps_rhand in meta["keypoints_right_hand"]:
            if each_kps_rhand is not None:
                kps_rhand.append([each_kps_rhand.x, each_kps_rhand.y])
                kps_rhand_p.append(each_kps_rhand.score)
            else:
                kps_rhand.append([None, None])
                kps_rhand_p.append(0.0)

        pose_meta.kps_rhand = np.array(kps_rhand)
        pose_meta.kps_rhand_p = np.array(kps_rhand_p)

        pose_metas.append(pose_meta)

    return pose_metas
