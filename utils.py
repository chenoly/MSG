import cv2
import bchlib
import numpy as np
from typing import Tuple
from skimage import img_as_float
from scipy.ndimage import gaussian_filter


class BCH:
    def __init__(self, BCH_POLYNOMIAL_=487, BCH_BITS_=5):
        self.bch = bchlib.BCH(BCH_POLYNOMIAL_, BCH_BITS_)

    def Encode(self, data_: bytearray):
        ecc = self.bch.encode(data_)
        packet = data_ + ecc
        packet_binary = ''.join(format(x, '08b') for x in packet)
        secret_ = [int(x) for x in packet_binary]
        return secret_

    def Decode(self, secret_: list):
        packet_binary = "".join([str(int(bit)) for bit in secret_])
        packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)
        data_, ecc = packet[:-self.bch.ecc_bytes], packet[-self.bch.ecc_bytes:]
        bit_flips = self.bch.decode(data_, ecc)
        if bit_flips[0] != -1:
            return bit_flips[1]
        return None



class Detector:
    def __init__(self, detect_num: int = 1, harris_block_size=3, k_size=3, k=0.04,
                 dilate_iterations: int = 4, corner_range: int = 32):
        """

        :param detect_num: the number to detect MSG
        :param harris_block_size: harris block size
        :param k_size: kernel size
        :param k: threshold of harris conner
        :param dilate_iterations: iterations to dilate
        :param corner_range: corrected range of detected points
        """
        self.harris_block_size = harris_block_size
        self.dilate_iterations = dilate_iterations
        self.detect_num = detect_num
        self.corner_range = corner_range
        self.k_size = k_size
        self.k = k

    def connectedComponentsRect(self, mask_image):
        """

        :param mask_image:
        :return:
        """
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_image, connectivity=8)
        connected_components = []
        for label in range(1, ret):
            left = stats[label, cv2.CC_STAT_LEFT]
            top = stats[label, cv2.CC_STAT_TOP]
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            component_pixels = np.where(labels == label)
            if len(component_pixels[0]) > 100:
                connected_components.append({
                    'left': left,
                    'top': top,
                    'width': width,
                    'height': height,
                    'pixel_count': len(component_pixels[0])
                })
        connected_components.sort(key=lambda x: x['pixel_count'], reverse=True)
        return connected_components

    def find_largest_inner_rectangle(self, harris_corner_mask, mask_pts):
        """

        :param harris_corner_mask:
        :param mask_pts:
        :return:
        """
        left = mask_pts['left']
        top = mask_pts['top']
        cc_r_width = mask_pts['width']
        cc_r_height = mask_pts['height']
        rec_mask = harris_corner_mask[top:top + cc_r_height, left:left + cc_r_width]
        contours = cv2.findContours(rec_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_inner_rectangle = None
        for contour in contours[1]:
            if len(contour) >= 4:
                area = cv2.contourArea(contour)
                if area > max_area:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) == 4:
                        max_area = area
                        max_inner_rectangle = approx
        if max_inner_rectangle is not None:
            max_inner_rectangle = max_inner_rectangle.reshape(4, 2).astype(np.float32)
            max_inner_rectangle[:, 0] = max_inner_rectangle[:, 0] + mask_pts['left']
            max_inner_rectangle[:, 1] = max_inner_rectangle[:, 1] + mask_pts['top']
            return max_inner_rectangle
        return None

    def find_largest_inner_rectangle_(self, gray_cc, cc_r_mask, target_size=(512, 512)):
        """

        :param target_size:
        :param gray_cc:
        :param cc_r_mask:
        :return:
        """
        contours = cv2.findContours(cc_r_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_inner_rectangle = None
        for contour in contours[1]:
            if len(contour) >= 4:
                area = cv2.contourArea(contour)
                if area > max_area:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) == 4:
                        max_area = area
                        max_inner_rectangle = approx
        if max_inner_rectangle is not None:
            max_inner_rectangle = max_inner_rectangle.reshape(4, 2).astype(np.float32)
            target_corners = np.array([[0, 0], [0, target_size[1]], [target_size[0], target_size[1]], [target_size[0], 0]], dtype=np.float32)
            perspective_matrix = cv2.getPerspectiveTransform(max_inner_rectangle, target_corners)
            transformed_image = cv2.warpPerspective(gray_cc, perspective_matrix, target_size)
            return transformed_image
        return None

    def find_corner_points(self, edge, pts: list, pts_center: list, corner_range):
        """

        :param pts_center:
        :param pts:
        :param edge:
        :param corner_range:
        :return:
        """
        pt_c_x, pt_c_y = pts_center
        pt_x, pt_y = pts
        dis_min = 1e9
        Height, Width = edge.shape
        final_x = pt_c_x
        final_y = pt_c_y
        for w in range(int(pt_c_x - corner_range / 2), int(pt_c_x + corner_range / 2)):
            for h in range(int(pt_c_y - corner_range / 2), int(pt_c_y + corner_range / 2)):
                if 0 <= w < Width and 0 <= h < Height:
                    pixel = edge[h, w]
                    if pixel == 255:
                        distance_ = np.abs(pt_x - w) + np.abs(pt_y - h)
                        if distance_ < dis_min:
                            dis_min = distance_
                            final_x = w
                            final_y = h
        return [final_x, final_y]

    def adjust_positioning_points(self, gray_image, pts, corner_range):
        """

        :param gray_image:
        :param pts:
        :param corner_range:
        :return:
        """
        gray_image_blur = cv2.GaussianBlur(gray_image, (7, 7), 0)
        edges = cv2.Canny(gray_image_blur, 50, 150)
        left_up_pt = self.find_corner_points(edges, pts[0], [pts[0][0] + int(corner_range / 2), pts[0][1] + int(corner_range / 2)], corner_range)
        right_up_pt = self.find_corner_points(edges, pts[1], [pts[1][0] - int(corner_range / 2), pts[1][1] + int(corner_range / 2)], corner_range)
        right_down_pt = self.find_corner_points(edges, pts[2], [pts[2][0] - int(corner_range / 2), pts[2][1] - int(corner_range / 2)], corner_range)
        left_down_pt = self.find_corner_points(edges, pts[3], [pts[3][0] + int(corner_range / 2), pts[3][1] - int(corner_range / 2)], corner_range)
        res_pts = np.asarray([left_up_pt, left_down_pt, right_down_pt, right_up_pt]).reshape(4, 2).astype(np.float32)
        return res_pts

    def calculate_perspective_transform(self, src_points, dst_points):
        """

        :param src_points:
        :param dst_points:
        :return:
        """
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        inverse_perspective_matrix = np.linalg.inv(perspective_matrix)
        return perspective_matrix, inverse_perspective_matrix

    def order_points(self, pts):
        """

        :param pts:
        :return:
        """
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def edge_detection_and_perspective_transform(self, gray_image, detect_pts, msg_size=(324, 324)):
        """

        :param detect_pts:
        :param gray_image:
        :param msg_size:
        :return:
        """
        detect_pts = self.order_points(detect_pts)
        src_points = self.adjust_positioning_points(gray_image, detect_pts, self.corner_range)
        dst_points = np.array([[0, 0], [0, msg_size[0]], [msg_size[1], msg_size[0]], [msg_size[1], 0]], dtype=np.float32)
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed_image = cv2.warpPerspective(gray_image, perspective_matrix, msg_size)
        return transformed_image, src_points

    def detect(self, gray_img, msg_size: Tuple):
        """

        :param gray_img:
        :param msg_size:
        :return:
        """
        result = []
        un_sharp_img = np.uint8(unsharp_mask(gray_img, 2, 4) * 255)
        corners = cv2.cornerHarris(un_sharp_img, self.harris_block_size, self.k_size, self.k)
        corner_distribution_map = np.zeros_like(un_sharp_img, dtype=np.uint8)
        corner_distribution_map[corners > 0.01 * corners.max()] = 255
        kernel = np.ones((6, 6), np.uint8)
        harris_corner_mask = cv2.dilate(corner_distribution_map, kernel, iterations=4)
        cc_rects = self.connectedComponentsRect(harris_corner_mask)[0:self.detect_num]
        if len(cc_rects) == 0:
            return result
        for cc_rect in cc_rects:
            detect_pts = self.find_largest_inner_rectangle(harris_corner_mask, cc_rect)
            if detect_pts is not None:
                res = self.edge_detection_and_perspective_transform(gray_img, detect_pts, msg_size)
                if res is not None:
                    warped_img, src_points = res
                    result.append([np.uint8(warped_img), src_points.tolist()])
        return result


def _unsharp_mask_single_channel(image, radius, amount, vrange):
    """Single channel implementation of the unsharp masking filter."""

    blurred = gaussian_filter(image,
                              sigma=radius,
                              mode='reflect')

    result = image + (image - blurred) * amount
    if vrange is not None:
        return np.clip(result, vrange[0], vrange[1], out=result)
    return result


def unsharp_mask(image, radius=1.0, amount=1.0, multichannel=False,
                 preserve_range=False):
    vrange = None  # Range for valid values; used for clipping.
    if preserve_range:
        fimg = image.astype(float)
    else:
        fimg = img_as_float(image)
        negative = np.any(fimg < 0)
        if negative:
            vrange = [-1., 1.]
        else:
            vrange = [0., 1.]

    if multichannel:
        result = np.empty_like(fimg, dtype=float)
        for channel in range(image.shape[-1]):
            result[..., channel] = _unsharp_mask_single_channel(
                fimg[..., channel], radius, amount, vrange)
        return result
    else:
        return _unsharp_mask_single_channel(fimg, radius, amount, vrange)
