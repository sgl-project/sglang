# -*- coding: utf-8 -*-

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

import io
import base64
import numpy as np
from PIL import Image


def to_html_frame(content):

    html_frame = f"""
    <html>
      <body>
        {content}
      </body>
    </html>
    """

    return html_frame


def to_single_row_table(caption: str, content: str):

    table_html = f"""
    <table border = "1">
        <caption>{caption}</caption>
        <tr>
            <td>{content}</td>
        </tr>
    </table>
    """

    return table_html


def to_image_embed_tag(image: np.ndarray):

    # Convert np.ndarray to bytes
    img = Image.fromarray(image)
    raw_bytes = io.BytesIO()
    img.save(raw_bytes, "PNG")

    # Encode bytes to base64
    image_base64 = base64.b64encode(raw_bytes.getvalue()).decode("utf-8")

    image_tag = f"""
    <img src="data:image/png;base64,{image_base64}" alt="Embedded Image">
    """

    return image_tag
