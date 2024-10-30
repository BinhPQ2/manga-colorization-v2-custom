import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from colorizator import MangaColorizator
from PIL import Image  # Import PIL for saving JPEG

def process_image(image, colorizator, size, denoiser, denoiser_sigma):
    colorizator.set_image(image, size, denoiser, denoiser_sigma)
    return colorizator.colorize()

def colorize_single_image(image_path, save_path, colorizator, size, denoiser, denoiser_sigma):
    image = plt.imread(image_path)
    colorization = process_image(image, colorizator, size, denoiser, denoiser_sigma)

    # Convert colorization to 8-bit unsigned integer for JPEG
    colorization = (colorization * 255).astype(np.uint8)

    # Save as JPEG using PIL
    Image.fromarray(colorization).convert('RGB').save(save_path, format='JPEG', quality=85)  # Ensure RGB format
    return True

def colorize_images(target_path, colorizator, path, size, denoiser, denoiser_sigma):
    images = os.listdir(path)
    
    for image_name in images:
        file_path = os.path.join(path, image_name)
        
        if os.path.isdir(file_path):
            continue
        
        name, ext = os.path.splitext(image_name)
        if ext.lower() not in ('.jpg', '.jpeg'):
            image_name = name + '.jpg'  # Change extension to .jpg
        
        print(f'Processing: {file_path}')
        
        save_path = os.path.join(target_path, image_name)
        colorize_single_image(file_path, save_path, colorizator, size, denoiser, denoiser_sigma)

def create_colorizer(gpu, generator, extractor):
    device = 'cuda' if gpu else 'cpu'
    return MangaColorizator(device, generator, extractor)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="Path to the image or directory")
    parser.add_argument("-gen", "--generator", default='networks/generator.zip', help="Path to the generator model")
    parser.add_argument("-ext", "--extractor", default='networks/extractor.pth', help="Path to the extractor model")
    parser.add_argument('-g', '--gpu', dest='gpu', action='store_true', help="Use GPU for processing")
    parser.add_argument('-nd', '--no_denoise', dest='denoiser', action='store_false', help="Disable denoising")
    parser.add_argument("-ds", "--denoiser_sigma", type=int, default=25, help="Denoiser sigma value")
    parser.add_argument("-s", "--size", type=int, default=576, help="Size for the colorization process")
    parser.set_defaults(gpu=False, denoiser=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    colorizer = create_colorizer(args.gpu, args.generator, args.extractor)
    
    if os.path.isdir(args.path):
        colorization_path = os.path.join(args.path, 'colorization')
        os.makedirs(colorization_path, exist_ok=True)
        colorize_images(colorization_path, colorizer, args.path, args.size, args.denoiser, args.denoiser_sigma)
        
    elif os.path.isfile(args.path):
        split = os.path.splitext(args.path)
        if split[1].lower() in ('.jpg', '.png', '.jpeg'):
            new_image_path = f"{split[0]}_colorized.jpg"  # Change extension to .jpg
            colorize_single_image(args.path, new_image_path, colorizer, args.size, args.denoiser, args.denoiser_sigma)
        else:
            print('Wrong image format')
    else:
        print('Wrong path format')

if __name__ == "__main__":
    main()
