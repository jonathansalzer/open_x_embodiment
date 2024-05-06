import pyzed.sl as sl
import cv2  # This is needed to save images in different formats

class Camera:
    def __init__(self):
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.camera_fps = 30

        self.err = self.zed.open(self.init_params)
        if self.err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        # Create an Image object to store the captured image
        self.image = sl.Mat()

    def __del__(self):
        self.zed.close()

    def get_picture(self):
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            image_ocv = self.image.get_data()
            return image_ocv
        print("Error: Could not grab image")
        return None

    def close(self):
        self.zed.close()