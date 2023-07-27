import torch
import cv2
import gc

# cheeky function to try and get the model to run more efficiently.
def resize_image(image, scale_percent=0.6):
    # Width and height re-scaling
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)

    # Resixe the image
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    return resized_image


# Context manager to REALLY reinforce cache clearing
class TorchCacheManager:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()
        gc.collect()
