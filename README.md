# Video Compression and Motion Analysis Project

## Project Description

This demonstration project shows how to compress and decompress individual video frames and analyze motion in real-time using optical flow. The project consists of the following steps:

1. **Image Division into Blocks:** The image is divided into smaller blocks of 8x8 pixels.
2. **Compression:** Each block undergoes Discrete Cosine Transform (DCT), quantization, and is then encoded using Run-Length Encoding (RLE).
3. **Decompression:** Compressed blocks are decoded, dequantized, and subjected to Inverse Discrete Cosine Transform (IDCT) to reconstruct the original image.
4. **Motion Analysis:** Optical flow between consecutive video frames is calculated, allowing for real-time motion visualization.

## Running the Project

To run the project, follow these steps:

1. Ensure that the camera is connected to the computer.
2. Run the `main.py` script:
    ```bash
    python main.py
    ```

## Requirements

- Python 3.x
- Libraries: numpy, opencv-python

## Repository Contents

- `main.py`: The main part of the program.
- `compression.py`: Module with compression and decompression functions.
