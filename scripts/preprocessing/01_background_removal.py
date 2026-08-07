from pathlib import Path
from rembg import remove, new_session
from PIL import Image


# Define input and output folders
INPUT_DIR = Path("../dataset_original")
OUTPUT_DIR = Path("../dataset_preprocessed")


# Create output folder if it does not exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Initialize U2-Net model session
session = new_session("u2net")


def remove_background(input_path, output_path):
    """
    Remove image background using U2-Net model.
    """

    with open(input_path, "rb") as input_file:
        input_data = input_file.read()

    output_data = remove(
        input_data,
        session=session
    )

    with open(output_path, "wb") as output_file:
        output_file.write(output_data)


def process_dataset():

    # Traverse all object folders
    for object_folder in INPUT_DIR.iterdir():

        if object_folder.is_dir():

            output_folder = OUTPUT_DIR / object_folder.name
            output_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            # Process all images
            for image_file in object_folder.glob("*"):

                if image_file.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png"
                ]:

                    output_file = (
                        output_folder /
                        f"{image_file.stem}_nobg.png"
                    )

                    remove_background(
                        image_file,
                        output_file
                    )

                    print(
                        f"Processed: {image_file.name}"
                    )


if __name__ == "__main__":

    process_dataset()

    print(
        "Background removal completed."
    )