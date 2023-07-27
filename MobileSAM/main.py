from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
import os
import matplotlib.pyplot as plt
import cv2
import torch
from timeit import default_timer


from util import TorchCacheManager, resize_image
from annotate import show_anns

if __name__ == "__main__":

    TIME = default_timer()

    # Clear cuda cache just incase
    torch.cuda.empty_cache()
    # Essentially reducing the page size but GPU
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

    # Configure our plot
    plt.figure(figsize=(20, 20))
    plt.axis('off')

    print(f'Finished Setup ({default_timer() - TIME:.2f}s)')

    with TorchCacheManager():
        # Read in our image
        image = cv2.imread(f'images/{input("Segment: images/")}')
        TIME = default_timer()
        # Convert it from BGR to RGB just incase
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize our image to see performance difference
        image = resize_image(image, 0.25)

    print(f'Read and preprocessed image ({default_timer() - TIME:.2f}s)')
    TIME = default_timer()

    with TorchCacheManager():
        # Defining our model type
        model_type = "vit_t"
        # Defining the path to our model checkpoint
        sam_checkpoint = "./weights/mobile_sam.pt"
        
        # If we have cuda available to us, use it
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Register our model
        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        
        print(f'Finished registering device ({default_timer() - TIME:.2f}s)')
        TIME = default_timer()

        # Define the device it's running on (CUDA or CPU)
        mobile_sam.to(device='cpu')

        print(f'Device: {device} enabled ({default_timer() - TIME:.2f}s)')
        TIME = default_timer()

        # Set the model to eval mode
        mobile_sam.eval()

    print(f'Model set to eval mode ({default_timer() - TIME:.2f}s)')
    TIME = default_timer()

    with TorchCacheManager():
        # Create our automatic mask generator (don't need to prompt it)
        mask_generator = SamAutomaticMaskGenerator(mobile_sam)

        print(f'Created Automatic Mask Generator ({default_timer() - TIME:.2f}s)')
        TIME = default_timer()
        
        print('Generating masks...')
        # Generate our masks
        masks = mask_generator.generate(image)

        print(f'Finished generating masks ({default_timer() - TIME:.2f}s)')
        
        # Display the original image
        plt.imshow(image)
        # Overlay the masks
        show_anns(masks)
        # Show the plot
        plt.show()
