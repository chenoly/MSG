import cv2
import argparse
import numpy as np
from code import MSG
from typing import Tuple
from utils import Detector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--msg_path', type=str, default='Images/captured_MSG.jpg')
    parser.add_argument('--correct_size', type=int, default=324)
    parser.add_argument('--N', type=int, default=36)
    args = parser.parse_args()

    save_corrected_msg_path = args.msg_path.split('.')[0] + "_corrected.png"
    save_located_msg_path = args.msg_path.split('.')[0] + "_located.png"
    model = MSG(args.N)
    detector = Detector()
    captured_msg = cv2.imread(args.msg_path, 0)
    result = detector.detect(captured_msg, (args.correct_size, args.correct_size))
    corrected_MSG = result[0][0]
    detected_pts = result[0][1]
    cv2.imwrite(save_corrected_msg_path, np.uint8(corrected_MSG))
    cv2.imwrite(save_located_msg_path, cv2.polylines(cv2.cvtColor(captured_msg, cv2.COLOR_GRAY2BGR), [np.int32(detected_pts)], isClosed=True, color=(0, 255, 0), thickness=5))

    ext_data, ext_bits = model.Decode(corrected_MSG)
    print("extracted data:", ext_data)
    print("extracted bits:", ext_bits)
    print("detected pts:", detected_pts)
