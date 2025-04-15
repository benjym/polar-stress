import os
import numpy as np
import json
from tqdm import tqdm
from glob import glob


def read_raw(filename, metadata):
    with open(filename, "rb") as f:
        # work out if it is 8bit or 16 bit
        eight_bit_file_size = metadata["width"] * metadata["height"] * 1
        actual_file_size = f.seek(0, 2)

        if actual_file_size == eight_bit_file_size:
            metadata["dtype"] = "uint8"
        elif actual_file_size == eight_bit_file_size * 2:
            metadata["dtype"] = "uint16"
        else:
            raise ValueError(
                f"File size does not match expected size for 8bit or 16bit data. Got {actual_file_size} bytes, expected {eight_bit_file_size} or {eight_bit_file_size * 2} bytes."
            )

        # f.seek(0)

        data = np.memmap(f, dtype=metadata["dtype"], mode="r", offset=0)
        data = data.reshape(
            (
                metadata["height"],
                metadata["width"],
            )
        )
        return data


def load_raw(foldername):
    json_file = foldername + "/recordingMetadata.json"
    if not os.path.exists(json_file):
        raise ValueError(f"Metadata file {json_file} does not exist. Please check the folder name.")

    frame_folder = foldername + "/0000000/"
    all_frames = glob(frame_folder + "frame*.raw")
    # frame_file = frame_folder + f"frame{str(0).zfill(10)}.raw"

    with open(json_file) as f:
        metadata = json.load(f)

    if len(all_frames) == 0:
        raise ValueError(f"No frames found in {frame_folder}.")
    else:
        # take median over all frames
        data = []
        for frame_file in tqdm(all_frames):
            data.append(read_raw(frame_file, metadata))
        data = np.median(np.array(data), axis=0)

    data = split_channels(data)

    return data, metadata


def split_channels(data):
    """
    Splits the data into its respective polarisation channels. Each superpixel is 4x4 pixels, and the channels are arranged in the following order:

    R_0 | R_45 | G1_0 | G1_45
    R_135 | R_90 | G1_135 | G1_90
    G2_0 | G2_45 | B_0 | B_45
    G2_135 | G2_90 | B_135 | B_90
    """

    # Reshape the data into a 4D array
    R_0 = data[0::4, 0::4]
    R_45 = data[0::4, 1::4]
    G1_0 = data[0::4, 2::4]
    G1_45 = data[0::4, 3::4]
    R_135 = data[1::4, 0::4]
    R_90 = data[1::4, 1::4]
    G1_135 = data[1::4, 2::4]
    G1_90 = data[1::4, 3::4]
    G2_0 = data[2::4, 0::4]
    G2_45 = data[2::4, 1::4]
    B_0 = data[2::4, 2::4]
    B_45 = data[2::4, 3::4]
    G2_135 = data[3::4, 0::4]
    G2_90 = data[3::4, 1::4]
    B_135 = data[3::4, 2::4]
    B_90 = data[3::4, 3::4]

    # Stack the channels into a 4D array
    I0 = np.stack((R_0, G1_0, G2_0, B_0), axis=-1)
    I90 = np.stack((R_90, G1_90, G2_90, B_90), axis=-1)
    I45 = np.stack((R_45, G1_45, G2_45, B_45), axis=-1)
    I135 = np.stack((R_135, G1_135, G2_135, B_135), axis=-1)

    # data is a 4D array with shape (height, width, colour, polarisation)
    data = np.stack(
        (
            I0,
            I90,
            I45,
            I135,
        ),
        axis=-1,
    )
    return data
