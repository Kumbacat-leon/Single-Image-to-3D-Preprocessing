import os
from PIL import Image


# Define project paths
INPUT_DIR = "../dataset_cropped"
OUTPUT_DIR = "../dataset_padded"


# Padding ratio
PADDING_RATIO = 0.2


def add_padding(image, padding_ratio):
    """
    Add extra space around the object.
    """

    width, height = image.size

    # Calculate padding size
    pad_w = int(width * padding_ratio)
    pad_h = int(height * padding_ratio)

    new_width = width + pad_w * 2
    new_height = height + pad_h * 2

    # Create transparent background
    padded_image = Image.new(
        "RGBA",
        (new_width, new_height),
        (0, 0, 0, 0)
    )

    # Paste original image into center
    padded_image.paste(
        image,
        (pad_w, pad_h),
        image
    )

    return padded_image



def process_dataset():

    objects = [
        "mouse",
        "bottle",
        "shoe"
    ]


    for obj in objects:

        input_folder = os.path.join(
            INPUT_DIR,
            obj
        )

        output_folder = os.path.join(
            OUTPUT_DIR,
            obj
        )


        os.makedirs(
            output_folder,
            exist_ok=True
        )


        for file in os.listdir(input_folder):

            if file.endswith(".png"):

                input_path = os.path.join(
                    input_folder,
                    file
                )


                output_path = os.path.join(
                    output_folder,
                    file.replace(
                        ".png",
                        "_pad.png"
                    )
                )


                image = Image.open(
                    input_path
                ).convert(
                    "RGBA"
                )


                padded = add_padding(
                    image,
                    PADDING_RATIO
                )


                padded.save(
                    output_path
                )


                print(
                    f"Processed: {output_path}"
                )


if __name__ == "__main__":

    process_dataset()