from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv

config_file = 'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = mmcv.imread(f'images/{input("Segment: images/")}')

# resize the image to the input size used during training
#img = mmcv.imresize(img, (512, 1024))

# Perform inference on the model and save the result
result = inference_model(model, img)

# Show the result using pyplot
show_result_pyplot(model, img, result, out_file='result.jpg')
