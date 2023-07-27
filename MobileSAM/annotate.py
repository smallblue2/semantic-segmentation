import numpy as np
import matplotlib.pyplot as plt

# Displays the masks on the plot
def show_anns(anns):
    if len(anns) == 0:
        return
    # Sort them so the larget mask by area is first
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # Get the current axes
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # Create a 4 channel image where the shape of the first two
    # dimensions match the shape of the largets mask, and the 
    # third dimension is 4 to represent RGBA channels.
    #
    # Initially set it to be fully transparent and white.
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    
    # Iterate over our masks, colouring them in as we go
    for ann in sorted_anns:
        # Grab the 'binary' mask
        mask = ann['segmentation']
        # Generate a 4-channel image array (random colour and 0.35
        # transparency)
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        # Apply this coloured mask to the original mask.
        img[mask] = color_mask

    # Display the image mask.
    ax.imshow(img)
