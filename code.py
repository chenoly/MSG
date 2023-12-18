import cv2
import utils
import numpy as np
from numpy import ndarray
from typing import Tuple, List, Any, Union


class MSG:
    def __init__(self, box_size, xi: float = 2.0, coh_i: int = 1, coh_j: int = 2):
        self.xi = xi
        self.coh_i = coh_i
        self.coh_j = coh_j
        self.K_x = None
        self.bch = utils.BCH()
        self.template_1 = None
        self.template_0 = None
        self.box_size = box_size
        self.InitParams()

    def InitParams(self):
        """

        :return:
        """
        self.K_x = np.zeros(shape=(self.box_size, self.box_size))
        self.template_1 = np.zeros(shape=(self.box_size, self.box_size))
        self.template_0 = np.zeros(shape=(self.box_size, self.box_size))
        for i in range(self.box_size):
            for j in range(self.box_size):
                b = (np.cos((2 * i + 1) * self.coh_i * np.pi / (2 * self.box_size)) * np.cos(
                    (2 * j + 1) * self.coh_j * np.pi / (2 * self.box_size)) -
                     np.cos((2 * i + 1) * self.coh_j * np.pi / (2 * self.box_size)) * np.cos(
                            (2 * j + 1) * self.coh_i * np.pi / (2 * self.box_size)))
                self.K_x[i, j] = b
                if b >= 0:
                    self.template_1[i, j] = 1
                else:
                    self.template_0[i, j] = 1

    def Encode(self, data: str, alpha: float = 0.7, gamma: float = 0.06):
        """

        :param data: data for embedding
        :param alpha: white-black pixel ratio
        :param gamma: factor for controlling embedding strength
        :return:
        """
        byte_array = bytearray(data.encode('utf-8'))
        bits = self.bch.Encode(byte_array)
        N = round(np.sqrt(len(bits)))
        com_N = N ** 2 - len(bits)
        bits_embed = bits + [0 for _ in range(com_N)]
        msg = np.zeros(shape=(self.box_size * N, self.box_size * N))
        I_CDP = np.random.binomial(1, alpha, size=(self.box_size * N, self.box_size * N))
        delta = gamma * min(abs(np.sum(self.template_1 * self.K_x)), abs(np.sum(self.template_0 * self.K_x)))
        index_bit = 0
        for h in range(0, self.box_size * N, self.box_size):
            for w in range(0, self.box_size * N, self.box_size):
                bit = bits_embed[index_bit]
                cdp_i = I_CDP[h:h + self.box_size, w:w + self.box_size]
                msg_block = self.compute_MSG_Block(cdp_i, bit, delta)
                msg[h:h + self.box_size, w:w + self.box_size] = msg_block
                index_bit += 1
        return np.uint8(msg * 255), bits_embed



    def compute_MSG(self, I_CDP: ndarray, G_1: ndarray, G_0: ndarray):
        """

        :param I_CDP: i-th CDP block
        :param G_1: optimization parameter for template bit one
        :param G_0: optimization parameter for template bit zero
        :return:
        """
        return I_CDP + self.template_1 * G_1 + self.template_0 * G_0

    def objective_function(self, w: int, delta: float, I_MSG: ndarray):
        """

        :param w: watermark bit
        :param delta: embedding strength
        :param I_MSG: the MSG block at t time
        :return:
        """
        delta_c = np.sum(I_MSG * self.K_x)
        if w == 1:
            f = (delta_c - delta) ** 2
        else:
            f = (delta_c + delta) ** 2
        return f

    def compute_gradients(self, w: int, delta: float, I_MSG: ndarray):
        """

        :param w: watermark bit
        :param delta: embedding strength
        :param I_MSG: the MSG block at t time
        :return:
        """
        hat_grad_G_1 = np.zeros(shape=(self.box_size, self.box_size))
        hat_grad_G_0 = np.zeros(shape=(self.box_size, self.box_size))
        if w == 1:
            grad_f_G1 = np.sign(2 * (np.sum(I_MSG * self.K_x) - delta) * self.template_1 * self.K_x)
            grad_f_G0 = np.sign(2 * (np.sum(I_MSG * self.K_x) - delta) * self.template_0 * self.K_x)
        else:
            grad_f_G1 = np.sign(2 * (np.sum(I_MSG * self.K_x) + delta) * self.template_1 * self.K_x)
            grad_f_G0 = np.sign(2 * (np.sum(I_MSG * self.K_x) + delta) * self.template_0 * self.K_x)

        condition = (-1) ** (w + 1) * np.sum(I_MSG * self.K_x) > delta
        if w == 1:
            xs_1, ys_1 = np.where((I_MSG == condition) & (self.template_1 == 1))
            xs_0, ys_0 = np.where((I_MSG != condition) & (self.template_0 == 1))
        else:
            xs_1, ys_1 = np.where((I_MSG != condition) & (self.template_1 == 1))
            xs_0, ys_0 = np.where((I_MSG == condition) & (self.template_0 == 1))

        if len(xs_1) > 0 and len(ys_1) > 0:
            index_1 = np.random.choice(range(len(xs_1)))
            grad1 = grad_f_G1[xs_1[index_1], ys_1[index_1]]
            hat_grad_G_1[xs_1[index_1], ys_1[index_1]] = grad1

        if len(xs_0) > 0 and len(ys_0) > 0:
            index_0 = np.random.choice(range(len(xs_0)))
            grad0 = grad_f_G0[xs_0[index_0], ys_0[index_0]]
            hat_grad_G_0[xs_0[index_0], ys_0[index_0]] = grad0
        return hat_grad_G_1, hat_grad_G_0

    def compute_MSG_Block(self, cdp_i, w, delta: float):
        """

        :param cdp_i: i-th CDP block
        :param delta: embedding strength
        :param w: watermark bit
        :return:
        """
        G_1 = np.zeros(shape=(self.box_size, self.box_size))
        G_0 = np.zeros(shape=(self.box_size, self.box_size))
        I_MSG = self.compute_MSG(cdp_i, G_1, G_0)
        f = self.objective_function(w, delta, I_MSG)
        while f > self.xi:
            I_MSG = self.compute_MSG(cdp_i, G_1, G_0)
            f = self.objective_function(w, delta, I_MSG)
            hat_grad_G_1, hat_grad_G_0 = self.compute_gradients(w, delta, I_MSG)
            G_1 -= hat_grad_G_1
            G_0 -= hat_grad_G_0
        I_MSG = self.compute_MSG(cdp_i, G_1, G_0)
        return I_MSG

    def Decode(self, captured_msg:ndarray) -> Tuple[str, list]:
        """

        :param captured_msg: captured corrected MSG
        :return:
        """
        ext_data, ext_bits = self.Extract_All(captured_msg)
        return ext_data, ext_bits

    def Extract_All(self, ac_code: ndarray) -> Tuple[Any, List[int]]:
        """

        :param ac_code: extract all watermark bits
        :return:
        """
        ext_bits = []
        code_size, _ = ac_code.shape
        for h in range(0, code_size, self.box_size):
            for w in range(0, code_size, self.box_size):
                start_h = h
                start_w = w
                end_h = start_h + self.box_size
                end_w = start_w + self.box_size
                code_block = ac_code[start_h:end_h, start_w:end_w]
                bit = self.Extract(code_block)
                ext_bits.append(bit)
        end_bit = int(len(ext_bits) / 8) * 8
        ext_data = self.bch.Decode(ext_bits[0:end_bit])
        return ext_data, ext_bits

    def Extract(self, ac_code_block: ndarray) -> int:
        """

        :param ac_code_block: extract one bit from MSG block
        :return:
        """
        d_c = np.sum(ac_code_block * self.K_x) * (2 / self.box_size)
        if d_c >= 0:
            return 1
        else:
            return 0
