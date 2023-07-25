import torch
from timeit import default_timer # for timing things
from sys import exit

# What model number are we using?
model_num = int(input('What model would you like to use?\n\n(1) DeepLabV3 + Resnet50\n(2) DeepLabV3 + Resnet101\n(3) DeepLabV3 + MobileNetV3 Large\n\nModel Number: '))
if model_num < 1 or model_num > 3:
    exit()

# Timer for timing operations performed in the script
TIME = default_timer()

# Get our model from pytorch hub
match model_num:
    case 1:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True) # 66.4 MIoU
    case 2: 
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)  # 67.4 MIoU
    case 3:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True) # 60.3 MIoU

print(f'Retrieved model... ({default_timer() - TIME:.2f}s)')
TIME = default_timer()

# Put the model in eval mode
model.eval() 

print(f'Model is in evaluation mode... ({default_timer() - TIME:.2f}s)')
TIME = default_timer()

# All pre-trained models expect images normalized in the same way - N, channel height width
# Where N is the number of images.

# The images must be:
# loaded within a range of [0, 1]
# normalized with mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]

# The model returns an OrderedDict with two Tensors of the same width and height as the input tensor
# but with 21 classes.

# output['out'] = semantic masks, output['aux'] = auxiliary loss values per pixel.
# 'out' is useful for inference - shape (N, 21, H, W)

filename = input('Image filename (leave empty to download example): ')

if filename == '':
    TIME = default_timer()
    # Download an example image
    import urllib
    url = "https://github.com/pytorch/hub/raw/master/images/deeplab1.png"
    filename = "deeplab1.png"
    
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    
    print(f'Downloaded sample image {filename} from {url} ({default_timer() - TIME:.2f}s)')
    TIME = default_timer()
else:
    TIME = default_timer()

# Inference

from PIL import Image
from torchvision import transforms
# Open and convert image to RGB
input_image = Image.open(filename)
input_image = input_image.convert("RGB")

print(f'Opened and converted image to RGB... ({default_timer() - TIME:.2f}s)')
TIME = default_timer()

# Create a preproccessor
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(f'Created preprocessor... ({default_timer() - TIME:.2f}s)')
TIME = default_timer()

# Preprocess the image
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # the model expects a mini-batch

print(f'Pre-processed the image... ({default_timer() - TIME:.2f}s)')
TIME = default_timer()

# Choose processing device
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
print(f'Cuda {"enabled" if torch.cuda.is_available() else "disabled"}... ({default_timer() - TIME:.2f}s)')

print('Running inference...')
TIME = default_timer()

# Run inference and get the output
with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

print(f'Ran inference... ({default_timer() - TIME:.2f}s)')

print('Displaying result')

# Create a color pallete, selecting a colour for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)

import matplotlib.pyplot as plt
# Create a subplot with 1 row and 2 columns
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# Display original input image on the left
axs[0].imshow(input_image)
axs[0].set_title('Input Image')
# Display output image on the right
axs[1].imshow(r)
axs[1].set_title('Output Segmentation')

# Create our legend:
# Create a legend for all classes
import matplotlib.patches as mpatches

# List our classes
classes = [
    "__background__",
    "aeroplane (1)",
    "bicycle (2)",
    "bird (3)",
    "boat (4)",
    "bottle (5)",
    "bus (6)",
    "car (7)",
    "cat (8)",
    "chair (9)",
    "cow (10)",
    "diningtable (11)",
    "dog (12)",
    "horse (13)",
    "motorbike (14)",
    "person (15)",
    "pottedplant (16)",
    "sheep (17)",
    "sofa (18)",
    "train (19)",
    "tvmonitor (20)",
]

# Create our handles
handles = [mpatches.Patch(color=colors[i] / 255., label=classes[i]) for i in range(21)]
axs[1].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Show the images
plt.show()
