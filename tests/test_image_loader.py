import unittest
import os
import numpy as np
from PIL import Image
from src.load_image import load_images_from_folder

class TestImageLoading(unittest.TestCase):
    def test_load_images_from_folder(self):
        folder_path = "../data-source/"
        loaded_images = load_images_from_folder(folder_path)
        
        # Check that the loaded_images list is not empty
        self.assertNotEqual(len(loaded_images), 0, "The list of loaded images is empty.")
