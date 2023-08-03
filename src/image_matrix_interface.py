from src.load_image import load_images_from_folder
from src.matrix_multiplication import matrix_multiplication_cuda
import numpy as np

def process_image_engine(folder_path):
    loaded_images = load_images_from_folder(folder_path)
    caculation_results = []
    for image_np_array in loaded_images:
        # generate a random np array with the same size as the image
        random_array = np.random.rand(image_np_array.shape[0], image_np_array.shape[1])
        print(f"image_np_array: {image_np_array}")
        caculation_results.append(matrix_multiplication_cuda(image_np_array, random_array))
    print("caculation_results: ", caculation_results)
    return caculation_results