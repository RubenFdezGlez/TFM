"""
    Package imports:

    os: File and directory management, including folder creation and file copying.
    PIL: Opens the image from the dataset to extract the original width and height.
    shutil: Copying image files to corresponding folders.
"""
import os
from PIL import Image
import shutil

class CCPDYOLOReorganizer:
    """Class for reorganizing the BDD100K dataset into YOLO format."""
    def __init__(self, images_path, dst_path): 
        """Initialize the reorganizer with paths.
        
        Args:
            images_path (str): Path to the original images path.
            dst_path (str): Path to the destination folder where the reorganized dataset will be stored.
        """
        self.images_path = images_path
        self.dst_path = dst_path

    def reorganize(self):
        os.makedirs(self.dst_path, exist_ok=True)

        for ccpd_folder in os.listdir(self.images_path):

            old_path = os.path.join(self.images_path, ccpd_folder)
            new_path = os.path.join(self.dst_path, ccpd_folder, "test")

            if os.path.isdir(old_path) and ccpd_folder.startswith("ccpd"):

                imgs_path = os.path.join(new_path, "images")
                labels_path = os.path.join(new_path, "labels")

                os.makedirs(imgs_path, exist_ok=True)
                os.makedirs(labels_path, exist_ok=True)

                for img in os.listdir(old_path):
                    if img.endswith((".jpg", ".jpeg", ".png")):
                        try:
                            img_path = os.path.join(old_path, img)
                            dst_path = os.path.join(imgs_path, img)

                            img_extension = img.split(".")[-1]
                            label_path = os.path.join(labels_path, img).replace(img_extension, "txt")

                            # Get the YOLO annotations
                            coords = img.split("-")[2]
                            up_left, bottom_right = coords.split("_")
                            ul_x, ul_y = up_left.split("&")
                            br_x, br_y = bottom_right.split("&")

                            im = Image.open(img_path)
                            img_width, img_height = im.size 

                            center_x = (int(br_x) + int(ul_x)) / 2 * img_width
                            center_y = (int(br_y) + int(ul_y)) / 2 * img_height

                            width = (int(br_x) - int(center_x)) / img_width
                            height = (int(br_y) - int(center_y)) / img_height

                            shutil.copy(img_path, dst_path)
                            with open(label_path, 'w') as f:
                                f.write(f"0 {center_x} {center_y} {width} {height}")
                        except Exception as e:
                            print(f"There was an exception: {e}")


if __name__ == '__main__':

    cc = CCPDYOLOReorganizer(
        images_path = "./../datasets/CCPD2019/",
        dst_path = "./../datasets/ccpd_yolo/"
    )

    cc.reorganize()