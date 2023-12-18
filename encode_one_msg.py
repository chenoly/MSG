import os
import cv2
import argparse
from code import MSG

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='Images/')
    parser.add_argument('--N', type=int, default=36)
    parser.add_argument('--data', type=str, default='hello')
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--gamma', type=float, default=0.06)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    model = MSG(args.N)
    generated_MSG, embedded_bits = model.Encode(args.data, args.alpha, args.gamma)
    cv2.imwrite(f"{args.save_path}/generated_MSG.png", generated_MSG)
    loaded_msg = cv2.imread(f"{args.save_path}/generated_MSG.png", 0)
    ext_data, ext_bits = model.Decode(loaded_msg)
    print("embedded bits: ", embedded_bits)
    print("extracted bits:", ext_bits)
    print("extracted data:", ext_data)
