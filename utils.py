
import numpy as np
from itertools import groupby
from pycocotools import mask as maskutil
from skimage import measure


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle
    


# def binary_mask_to_polygon(binary_mask, tolerance=0):
#     """Converts a binary mask to COCO polygon representation
#     Args:
#         binary_mask: a 2D binary numpy array where '1's represent the object
#         tolerance: Maximum distance from original points of polygon to approximated
#             polygonal chain. If tolerance is 0, the original coordinate array is returned.
#     """
#     polygons = []
#     # pad mask to close contours of shapes which start and end at an edge
#     padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
#     contours = measure.find_contours(padded_binary_mask, 0.5)
#     contours = np.subtract(contours, 1)
#     for contour in contours:
#         contour = close_contour(contour)
#         contour = measure.approximate_polygon(contour, tolerance)
#         if len(contour) < 3:
#             continue
#         contour = np.flip(contour, axis=1)
#         segmentation = contour.ravel().tolist()
#         # after padding and subtracting 1 we may get -0.5 points in our segmentation 
#         segmentation = [0 if i < 0 else i for i in segmentation]
#         polygons.append(segmentation)

#     return polygons

