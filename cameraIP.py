import cv2
import time 
import numpy as np
import math
from compression import divide_into_blocks, compress, decompress, merge_blocks

BLOCK_SIZE = 8

def generate_grid_coordinates(step, h, w):
    y_coords = []
    x_coords = []

    for y in range(step//2, h, step):
        for x in range(step//2, w, step):
            y_coords.append(y)
            x_coords.append(x)

    return y_coords, x_coords

def motion_flow(img_source, flow, threshold=2.0, step=16):
    h, w = img_source.shape[:2]
    y_coords, x_coords = generate_grid_coordinates(step, h, w)
    fx = []
    fy = []
    for y, x in zip(y_coords, x_coords):
        fx.append(flow[y, x][0])
        fy.append(flow[y, x][1])

    lines = []
    for x, y, dx, dy in zip(x_coords, y_coords, fx, fy):
        if math.sqrt(dx ** 2 + dy ** 2) > threshold:
            lines.append([x, y, int(x - dx), int(y - dy)])
    lines = np.array(lines)
    lines = lines.reshape(-1, 2, 2)
    img_grscale = cv2.cvtColor(img_source, cv2.COLOR_GRAY2BGR)
    for line in lines:
        cv2.polylines(img_grscale, [line], 0, (0, 255, 0))

    return img_grscale

def apply_filter(img, kernel_size=(5, 5)):
    kh, kw = kernel_size
    pad_height = kh // 2
    pad_width = kw // 2

    img_padded = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    filtered_img = np.zeros_like(img, dtype=np.float64)

    # print(time.time())
    # start = time.time()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            filtered_img[i, j] = np.mean(img_padded[
                i:i + kernel_size[0],
                j:j + kernel_size[1]
            ])

    end = time.time()
    # print("czas konca: ", end-start)

    filtered_img = np.uint8(filtered_img)

    return filtered_img


def apply_filter_smooth(img):
    filtered_img = cv2.blur(img, (5, 5))

    return filtered_img


def main():
    cap = cv2.VideoCapture(0)

    target_width = 320
    target_height = 240

    suc, prev = cap.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prevgray = cv2.resize(prevgray, (target_width, target_height))

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (target_width, target_height))

        blocks = divide_into_blocks(gray, BLOCK_SIZE)
        compressed_blocks = compress(blocks)
        decompressed_blocks = decompress(compressed_blocks, BLOCK_SIZE)
        reconstructed_image = merge_blocks(decompressed_blocks, gray.shape)

        # filtered_gray = apply_filter_smooth(reconstructed_image)
        flow = cv2.calcOpticalFlowFarneback(prevgray, reconstructed_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = reconstructed_image

        cv2.imshow('frame', motion_flow(gray, flow))
        # cv2.imshow('Original', prev)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()