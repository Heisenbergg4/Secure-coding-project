from matplotlib import pyplot as plt
import numpy as np
import datetime
import pickle
import os
import sys

# Make a list of possible sizes if not yet in the project folder
if not os.path.exists("./sizes.pkl"):
    sizes = []
    for i in range(1, 1_000_000):
        size = (i**2) * 3
        sizes.append(size)
    with open("sizes.pkl", "wb") as f:
        pickle.dump(sizes, f)

# Load the list of possible square image sizes for files up to 3TB, for passing into findNearestUnder
with open("sizes.pkl", "rb") as f:
    size_list = pickle.load(f)

# Select the nearest list length that satisfies dimensions for a square image [W,H,3] without exceeding
def findNearestUnder(size, size_list):
    for i in range(len(size_list)):
        if (size > size_list[i]):
            continue
        return (size_list[i-1], i)

# Loop through the categories (benign and malicious), create .png files from them, and save into data/output/
def exeToImg(input_file, output_folder, interpolation='nearest', dpi=300):
    # Create output paths for saving images directly to the desired output folder
    static_output_folder = os.path.join("static", "output", "all_images")
    os.makedirs(static_output_folder, exist_ok=True)

    print(f"Converting {input_file} to tensor...")

    # Read the file as a byte array
    with open(input_file, 'rb') as f:
        byte_array = np.fromfile(f, dtype=np.uint8)
        size = findNearestUnder(byte_array.size, size_list)
        pic = np.reshape(byte_array[:size[0]], (size[1], size[1], 3))

        # Append characteristics to a log file called data.csv
        with open(os.path.join(output_folder, 'data.csv'), 'a') as f:
            f.write("\n{},{},{},{},{},{},{},{}".format(
                os.path.basename(input_file), datetime.datetime.now(), interpolation,
                dpi, byte_array.size, size[0], byte_array.size - size[0],
                (byte_array.size - size[0]) / byte_array.size
            ))

    print(f"Drawing as an image with interpolation: {interpolation} and dpi: {dpi}...")
    plt.imshow(pic, interpolation=interpolation)

    # Save image directly to the static output folder
    image_filename = os.path.splitext(os.path.basename(input_file))[0] + f'_{interpolation}_{dpi}.png'
    plt.savefig(os.path.join(static_output_folder, image_filename), bbox_inches='tight', dpi=dpi)

    print(f"Saved image to {static_output_folder}/{image_filename}")
    plt.clf()  # Clear the current figure

# Set interpolation, dpi, input location, and output location
def process_exe_file(input_file):
    interps = ['nearest', 'lanczos']
    dpis = [120, 300, 600, 1200]

    # Loop through the options chosen above to create image files
    for interp in interps:
        for dpi in dpis:
            exeToImg(input_file, 'static/output', interpolation=interp, dpi=dpi)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exe_file_path = sys.argv[1]  # Get the path of the exe file from the command line argument
        process_exe_file(exe_file_path)
    else:
        print("No input file provided.")
