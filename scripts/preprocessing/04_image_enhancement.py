import os
import cv2
import numpy as np


# Define input and output folders
INPUT_DIR = "../dataset_padded"
OUTPUT_DIR = "../dataset_enhanced"


# CLAHE parameters
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)


def apply_clahe(image):
    """
    Apply CLAHE contrast enhancement.
    """

    # Convert image from BGR to LAB color space
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    # Split LAB channels
    l, a, b = cv2.split(lab)


    # Create CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=CLIP_LIMIT,
        tileGridSize=TILE_GRID_SIZE
    )


    # Enhance the luminance channel
    enhanced_l = clahe.apply(l)


    # Merge channels back
    enhanced_lab = cv2.merge(
        [enhanced_l, a, b]
    )


    # Convert back to BGR
    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )


    return enhanced_image



def apply_sharpening(image):
    """
    Apply image sharpening using unsharp masking.
    """

    # Create blurred version
    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        3
    )


    # Combine original and blurred image
    sharpened = cv2.addWeighted(
        image,
        1.5,
        blurred,
        -0.5,
        0
    )


    return sharpened



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


        # Create output folder
        os.makedirs(
            output_folder,
            exist_ok=True
        )


        for filename in os.listdir(input_folder):

            if filename.endswith(".png"):

                input_path = os.path.join(
                    input_folder,
                    filename
                )


                output_path = os.path.join(
                    output_folder,
                    filename.replace(
                        ".png",
                        "_enhanced.png"
                    )
                )


                # Read image
                image = cv2.imread(
                    input_path,
                    cv2.IMREAD_COLOR
                )


                if image is None:
                    continue


                # Apply CLAHE
                enhanced = apply_clahe(
                    image
                )


                # Apply sharpening
                enhanced = apply_sharpening(
                    enhanced
                )


                # Save result
                cv2.imwrite(
                    output_path,
                    enhanced
                )


                print(
                    f"Processed: {output_path}"
                )



if __name__ == "__main__":

    process_dataset()