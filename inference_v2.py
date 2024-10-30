import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from colorizator import MangaColorizator
from PIL import Image
import torch  # Import PyTorch to check for GPU availability

def process_image(image, colorizator, size, denoiser, denoiser_sigma):
    colorizator.set_image(image, size, denoiser, denoiser_sigma)
    return colorizator.colorize()

def colorize_single_image(image_path, save_path, colorizator, size, denoiser, denoiser_sigma):
    image = plt.imread(image_path)
    colorization = process_image(image, colorizator, size, denoiser, denoiser_sigma)
    colorization = (colorization * 255).astype(np.uint8)
    Image.fromarray(colorization).convert('RGB').save(save_path, format='JPEG')
    return True

def colorize_images(target_path, colorizator, path, size, denoiser, denoiser_sigma):
    images = os.listdir(path)
    for image_path in images:
        file_path = os.path.join(path, image_path)
        if os.path.isdir(file_path):
            continue
        image_name, image_ext = os.path.splitext(image_path)
        if image_ext.lower() not in ('.jpg', '.jpeg'):
            image_path = image_name + '.jpg'
        print(f'Processing: {file_path}')
        save_path = os.path.join(target_path, image_path)
        colorize_single_image(file_path, save_path, colorizator, size, denoiser, denoiser_sigma)

def create_colorizer(device, generator, denoiser):
    return MangaColorizator(device, generator, denoiser)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="Path to the image or directory")
    parser.add_argument("-gen", "--generator", default='networks/generator.zip', help="Path to the generator model")
    parser.add_argument("-des_path", "--denoiser_path", default='denoising/models/', help="Path to the denoiser model")
    parser.add_argument('-g', '--gpu', dest='gpu', action='store_true', help="Force usage of GPU")
    parser.add_argument('-nd', '--no_denoise', dest='denoiser', action='store_false', help="Disable denoising")
    parser.add_argument("-ds", "--denoiser_sigma", type=int, default=25, help="Denoiser sigma value")
    parser.add_argument("-s", "--size", type=int, default=576, help="Size for the colorization process")
    parser.set_defaults(gpu=False, denoiser=True)
    return parser.parse_args()

def main():
    args = parse_args()

    # Set the device: use GPU if available and not forced to use CPU
    device = 'cuda' if args.gpu or (torch.cuda.is_available() and not args.gpu) else 'cpu'
    
    colorizer = create_colorizer(device, args.generator, args.denoiser_path)
    
    if os.path.isdir(args.path):
        colorization_path = os.path.join(args.path, 'colorization')
        os.makedirs(colorization_path, exist_ok=True)
        colorize_images(colorization_path, colorizer, args.path, args.size, args.denoiser, args.denoiser_sigma)
        
    elif os.path.isfile(args.path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        image_name, image_ext = os.path.splitext(args.path)
        if image_ext.lower() in image_extensions:
            new_image_path = f"{image_name}_colorized.jpg"
            colorize_single_image(args.path, new_image_path, colorizer, args.size, args.denoiser, args.denoiser_sigma)
        else:
            print('Wrong image format')
    else:
        print('Wrong path format')

if __name__ == "__main__":
    main()
