from src.load_image import load_images_from_folder
from src.matrix_multiplication import matrix_multiplication_cuda
import numpy as np

def process_image_engine(folder_path):
    loaded_images = load_images_from_folder(folder_path)
    print("Loaded images: {}".format(loaded_images.shape))
    random_array = np.random.rand(loaded_images.shape[1], loaded_images.shape[0])
    print("Random array: {}".format(random_array.shape))
    return matrix_multiplication_cuda(loaded_images, random_array)