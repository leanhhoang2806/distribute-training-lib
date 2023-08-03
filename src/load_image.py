from PIL import Image
import os
import numpy as np

def load_images_from_folder(folder_path):
    image_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image_path = os.path.join(folder_path, filename)
            image_pil = Image.open(image_path)
            image_np = np.array(image_pil)
            image_list.append(image_np)
    return image_list

# Provide the folder path containing your images
folder_path = "/app/data-source/"
loaded_images = load_images_from_folder(folder_path)

# Now, loaded_images is a list of NumPy arrays, each representing an image from the folder
# You can access individual images by indexing the list, e.g., loaded_images[0], loaded_images[1], etc.
