import unittest
import os
import numpy as np
from PIL import Image
from src.image_matrix_interface import process_image_engine
import os

def count_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError("The specified folder does not exist.")
    
    files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    return len(files)


class TestProcessImageEngine(unittest.TestCase):
    def setUp(self):
        # Create a temporary folder with sample image files for testing
        self.test_folder = "test_images"
        os.makedirs(self.test_folder, exist_ok=True)

        # Create sample images (size: 100x100) for testing
        for i in range(5):
            image_np = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            image = Image.fromarray(image_np)
            image.save(os.path.join(self.test_folder, f"image{i}.png"))

    def tearDown(self):
        # Remove the temporary test folder and images
        for filename in os.listdir(self.test_folder):
            os.remove(os.path.join(self.test_folder, filename))
        os.rmdir(self.test_folder)

    def test_process_image_engine(self):
        folder_path = '/app/data-source/'
        calculation_results =  process_image_engine(folder_path)
        self.assertNotEqual(len(calculation_results), 0, "The list of calculation results is empty.")
        self.assertEqual(len(calculation_results), count_files_in_folder(folder_path), "number of calculation results is not equal to number of images in folder")

