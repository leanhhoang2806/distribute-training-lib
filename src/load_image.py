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
            # flatten the image
            image_np = image_np.reshape(-1)
            
            image_list.append(np.transpose(image_np))
    return image_list
