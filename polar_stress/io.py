import numpy as np
import json
import polar_stress.plotting


def load_raw(foldername):
    json_file = foldername + "/recordingMetadata.json"
    frame_folder = foldername + "/0000000/"
    frame_file = frame_folder + f"frame{str(0).zfill(10)}.raw"

    with open(json_file) as f:
        metadata = json.load(f)

    with open(frame_file, "rb") as f:
        data = np.memmap(f, dtype="uint8", mode="r", offset=0)
        data = data.reshape(
            (
                metadata["height"],
                metadata["width"],
                3,
            )
        )

    return data, metadata


def split_channels(data2D):
    """
    Splits the data into its respective polarisation channels.
    """
    I0R = data2D[::2, ::2]
    I90R = data2D[1::2, 1::2]
    I45R = data2D[::2, 1::2]
    I135R = data2D[1::2, ::2]

    return I0R, I90R, I45R, I135R


if __name__ == "__main__":
    import sys

    foldername = sys.argv[1]
    data, metadata = load_raw(foldername)
    polar_stress.plotting.show_all_channels(data)
