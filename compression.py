import numpy as np
import math
import cv2

QUANTIZATION_MATRIX_WIKIPEDIA = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

QUANTIZATION_MATRIX_STANFORD = np.array([
    [3, 5, 7, 9, 11, 13, 15, 17],
    [5, 7, 9, 11, 13, 15, 17, 19],
    [7, 9, 11, 13, 15, 17, 19, 21],
    [9, 11, 13, 15, 17, 19, 21, 23],
    [11, 13, 15, 17, 19, 21, 23, 25],
    [13, 15, 17, 19, 21, 23, 25, 27],
    [15, 17, 19, 21, 23, 25, 27, 29],
    [17, 19, 21, 23, 25, 27, 29, 31]
])

def divide_into_blocks(image, block_size):
    h, w = image.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            blocks.append(block)
    return blocks

def merge_blocks(blocks, image_shape):
    h, w = image_shape
    image = np.zeros((h, w))
    idx = 0
    block_size = blocks[0].shape[0]
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            image[i:i+block_size, j:j+block_size] = blocks[idx]
            idx += 1
    return image

def dct_2d(block):
    M, N = block.shape
    dct_result = np.zeros((M, N))

    for u in range(M):
        for v in range(N):
            sum_val = 0.0
            for x in range(M):
                for y in range(N):
                    alpha_u = math.sqrt(1/M) if u == 0 else math.sqrt(2/M)
                    alpha_v = math.sqrt(1/N) if v == 0 else math.sqrt(2/N)
                    sum_val += alpha_u * alpha_v * block[x, y] * math.cos((math.pi * u / M) * (x + 0.5)) * math.cos((math.pi * v / N) * (y + 0.5))
            dct_result[u, v] = sum_val

    return dct_result

def idct_2d(dct_coeffs):
    M, N = dct_coeffs.shape
    reconstructed_block = np.zeros((M, N))

    for x in range(M):
        for y in range(N):
            sum_val = 0.0
            for u in range(M):
                for v in range(N):
                    alpha_u = math.sqrt(1/M) if u == 0 else math.sqrt(2/M)
                    alpha_v = math.sqrt(1/N) if v == 0 else math.sqrt(2/N)
                    sum_val += alpha_u * alpha_v * dct_coeffs[u, v] * math.cos((math.pi * u / M) * (x + 0.5)) * math.cos((math.pi * v / N) * (y + 0.5))
            reconstructed_block[x, y] = sum_val

    return reconstructed_block

def quantize(block, quant_matrix):
    return np.round(block / quant_matrix, 1)

def dequantize(block, quant_matrix):
    return block * quant_matrix

def zigzag_order(block):
    zigzag = []
    for i in range(2 * block.shape[0] - 1):
        if i % 2 == 0:
            for j in range(i + 1):
                if j < block.shape[0] and (i - j) < block.shape[1]:
                    zigzag.append(block[j, i - j])
        else:
            for j in range(i + 1):
                if j < block.shape[0] and (i - j) < block.shape[1]:
                    zigzag.append(block[i - j, j])
    return zigzag

def inverse_zigzag_order(zigzag, block_size):
    block = np.zeros((block_size, block_size))
    index = 0
    for i in range(2 * block_size - 1):
        if i % 2 == 0:
            for j in range(i + 1):
                if j < block_size and (i - j) < block_size:
                    block[j, i - j] = zigzag[index]
                    index += 1
        else:
            for j in range(i + 1):
                if j < block_size and (i - j) < block_size:
                    block[i - j, j] = zigzag[index]
                    index += 1
    return block

def rle_encode(data):
    encoding = []
    prev_char = data[0]
    count = 1

    for char in data[1:]:
        if char == prev_char:
            count += 1
        else:
            encoding.append((prev_char, count))
            prev_char = char
            count = 1

    encoding.append((prev_char, count))
    return encoding

def rle_decode(data):
    decoding = []

    for char, count in data:
        decoding.extend([char] * count)

    return decoding

def compress(blocks):
    dct_blocks = []
    for block in blocks:
        dct_block = cv2.dct(np.float32(block))
        # dct_block = dct_2d(block)
        quantized_block = quantize(dct_block, QUANTIZATION_MATRIX_STANFORD)
        zigzagged_block = zigzag_order(quantized_block)
        rle_encoded_block = rle_encode(zigzagged_block)
        dct_blocks.append(rle_encoded_block)
    return dct_blocks

def decompress(dct_blocks, block_size):
    reconstructed_blocks = []
    for rle_encoded_block in dct_blocks:
        zigzagged_block = rle_decode(rle_encoded_block)
        quantized_block = inverse_zigzag_order(zigzagged_block, block_size)
        dequantized_block = dequantize(quantized_block, QUANTIZATION_MATRIX_STANFORD)
        reconstructed_block = cv2.idct(np.float32(dequantized_block))
        # reconstructed_block = idct_2d(dequantized_block)
        reconstructed_blocks.append(reconstructed_block)
    return reconstructed_blocks

def save_encoded_data_to_file(encoded_data, file_path):
    with open(file_path, 'w') as file:
        for block in encoded_data:
            for char, count in block:
                file.write(f"{char}:{count} ")
            file.write("\n")

def read_image_to_array(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Nie można wczytać obrazu z podanej ścieżki.")
        return None

    return image

def main():
    image_path = "zdjecie2.png"
    decompressed_image_path = "decompressed_zdjecie2.jpeg"
    image_matrix = read_image_to_array(image_path)

    if image_matrix is not None:
        print("Obraz wczytano pomyślnie.")

    block_size = 8
    blocks = divide_into_blocks(image_matrix, block_size)

    compressed_blocks = compress(blocks)

    decompressed_blocks = decompress(compressed_blocks, block_size)
    reconstructed_image = merge_blocks(decompressed_blocks, image_matrix.shape)

    cv2.imwrite(decompressed_image_path, np.uint8(reconstructed_image))

    cv2.imshow('Oryginalny obraz', image_matrix)
    cv2.imshow('Odzyskany obraz', np.uint8(reconstructed_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()