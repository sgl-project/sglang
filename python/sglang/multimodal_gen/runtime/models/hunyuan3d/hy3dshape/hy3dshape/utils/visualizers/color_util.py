# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import numpy as np
import matplotlib.pyplot as plt


# Helper functions
def get_colors(inp, colormap="viridis", normalize=True, vmin=None, vmax=None):
    colormap = plt.cm.get_cmap(colormap)
    if normalize:
        vmin = np.min(inp)
        vmax = np.max(inp)

    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))[:, :3]


def gen_checkers(n_checkers_x, n_checkers_y, width=256, height=256):
    # tex dims need to be power of two.
    array = np.ones((width, height, 3), dtype='float32')

    # width in texels of each checker
    checker_w = width / n_checkers_x
    checker_h = height / n_checkers_y

    for y in range(height):
        for x in range(width):
            color_key = int(x / checker_w) + int(y / checker_h)
            if color_key % 2 == 0:
                array[x, y, :] = [1., 0.874, 0.0]
            else:
                array[x, y, :] = [0., 0., 0.]
    return array


def gen_circle(width=256, height=256):
    xx, yy = np.mgrid[:width, :height]
    circle = (xx - width / 2 + 0.5) ** 2 + (yy - height / 2 + 0.5) ** 2
    array = np.ones((width, height, 4), dtype='float32')
    array[:, :, 0] = (circle <= width)
    array[:, :, 1] = (circle <= width)
    array[:, :, 2] = (circle <= width)
    array[:, :, 3] = circle <= width
    return array

