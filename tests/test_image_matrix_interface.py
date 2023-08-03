import unittest
from src.image_matrix_interface import load_images_from_folder, process_image_engine

class TestImageProcessing(unittest.TestCase):

    def setUp(self):
        # Create a temporary folder and add some test images
        self.test_folder = "/app/data-source"

    def test_load_images_from_folder(self):
        # Test loading and flattening the images from the folder
        loaded_images = load_images_from_folder(self.test_folder)
        self.assertTrue(loaded_images.shape[0] > 0)

    def test_process_image_engine(self):
        # Test processing the images from the folder using process_image_engine
        output = process_image_engine(self.test_folder)
        self.assertIsNotNone(output)
