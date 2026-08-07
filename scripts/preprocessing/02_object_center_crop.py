from pathlib import Path
from PIL import Image
import numpy as np


# Define input and output folders
INPUT_DIR = Path("../dataset_preprocessed")
OUTPUT_DIR = Path("../dataset_cropped")


# Create output directory
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def crop_object(image_path, output_path, padding=20):
    """
    Crop object area based on alpha channel.
    """

    image = Image.open(image_path).convert("RGBA")

    # Extract alpha channel
    alpha = np.array(image)[:, :, 3]

    # Find non-transparent pixels
    coords = np.where(alpha > 0)

    if len(coords[0]) == 0:
        print(f"No object detected: {image_path.name}")
        return

    # Calculate bounding box
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()


    # Add padding around object
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)

    x_max = min(
        image.width,
        x_max + padding
    )

    y_max = min(
        image.height,
        y_max + padding
    )


    # Crop image
    cropped = image.crop(
        (
            x_min,
            y_min,
            x_max,
            y_max
        )
    )


    # Save cropped image
    cropped.save(output_path)



def process_dataset():

    # Traverse object folders
    for object_folder in INPUT_DIR.iterdir():

        if object_folder.is_dir():

            output_folder = (
                OUTPUT_DIR /
                object_folder.name
            )

            output_folder.mkdir(
                parents=True,
                exist_ok=True
            )


            # Process images
            for image_file in object_folder.glob("*.png"):

                output_file = (
                    output_folder /
                    f"{image_file.stem}_crop.png"
                )


                crop_object(
                    image_file,
                    output_file
                )


                print(
                    f"Cropped: {image_file.name}"
                )


if __name__ == "__main__":

    process_dataset()

    print(
        "Object-centred cropping completed."
    )