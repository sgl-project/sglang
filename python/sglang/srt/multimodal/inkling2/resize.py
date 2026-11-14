import math

import numpy as np
from numba import njit


@njit(cache=True)
def _sinc(value):
    if value == np.float32(0):
        return np.float32(1)
    angle = value * np.float32(math.pi)
    return np.float32(math.sin(angle)) / angle


@njit(cache=True)
def _weights(source_size, target_size, output_index):
    ratio = np.float32(source_size) / np.float32(target_size)
    scale = max(np.float32(1), ratio)
    support = np.float32(3) * scale
    center = (np.float32(output_index) + np.float32(0.5)) * ratio
    left = min(max(math.floor(center - support), 0), source_size - 1)
    right = min(max(math.ceil(center + support), left + 1), source_size)
    center -= np.float32(0.5)
    weights = np.empty(right - left, dtype=np.float32)
    total = np.float32(0)
    for index in range(left, right):
        distance = (np.float32(index) - center) / scale
        weight = np.float32(0)
        if abs(distance) < np.float32(3):
            weight = _sinc(distance) * _sinc(distance / np.float32(3))
        weights[index - left] = weight
        total += weight
    for index in range(right - left):
        weights[index] /= total
    return left, weights


@njit(cache=True)
def resize_lanczos3(pixels, width, height):
    source_height, source_width, channels = pixels.shape
    vertical = np.empty((height, source_width, channels), dtype=np.float32)
    for y in range(height):
        left, weights = _weights(source_height, height, y)
        for x in range(source_width):
            for channel in range(channels):
                value = np.float32(0)
                for offset in range(len(weights)):
                    value += (
                        np.float32(pixels[left + offset, x, channel]) * weights[offset]
                    )
                vertical[y, x, channel] = value
    output = np.empty((height, width, channels), dtype=np.uint8)
    for x in range(width):
        left, weights = _weights(source_width, width, x)
        for y in range(height):
            for channel in range(channels):
                value = np.float32(0)
                for offset in range(len(weights)):
                    value += vertical[y, left + offset, channel] * weights[offset]
                value = min(max(value, np.float32(0)), np.float32(255))
                output[y, x, channel] = math.floor(float(value) + 0.5)
    return output
