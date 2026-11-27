# Adapted from LingBot-Video (https://github.com/Robbyant/lingbot-video).
#
# SPDX-License-Identifier: Apache-2.0

import PIL.Image


def condition_frame(image: PIL.Image.Image | list[PIL.Image.Image]) -> PIL.Image.Image:
    if isinstance(image, list) and len(image) == 1:
        image = image[0]
    if not isinstance(image, PIL.Image.Image):
        raise ValueError("LingBot-Video TI2V requires exactly one PIL reference frame")
    return image
