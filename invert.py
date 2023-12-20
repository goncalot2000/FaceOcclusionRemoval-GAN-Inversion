import sys
import os
import argparse
from tqdm import tqdm
import numpy as np
import random
import torch
import warnings

import cv2
import itertools

from utils.inverter import StyleGANInverter
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image

from models.stylegan_generator import StyleGANGenerator
from models.stylegan_encoder import StyleGANEncoder
from models.perceptual_model import PerceptualModel

from models.styleNAT_discriminator import Discriminator

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Name of the GAN model.')
    parser.add_argument('dataset', type=str,
                      help='Input Directory with the reference images, occluded images and occlusion masks.')   
    parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/inversion/${IMAGE_LIST}` '
                           'will be used by default.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate for optimization. (default: 0.01)')
    parser.add_argument('--num_iterations', type=int, default=100,
                      help='Number of optimization iterations. (default: 100)')
    parser.add_argument('--num_results', type=int, default=5,
                      help='Number of intermediate optimization results to '
                           'save for each sample. (default: 5)')
    parser.add_argument('--loss_weight_feat', type=float, default=5e-5,
                      help='The perceptual loss scale for optimization. '
                           '(default: 5e-5)')
    parser.add_argument('--loss_weight_enc', type=float, default=2.0,
                      help='The encoder loss scale for optimization.'
                           '(default: 2.0)')
    parser.add_argument('--loss_discriminator', type=float, default=0.0,
                      help='The discriminator loss for optimization.'
                           '(default: 0.0), if you want to use it 0.1 was the best value obtained.')
    parser.add_argument('--prior_flag', type=float, default=1.0,
                      help='Prior knowledge flag for optimization.'
                           '(default: 1.0)')
    parser.add_argument('--proportional_shift', type=float, default=13,
                      help='Proportional shift for the sigmoid weighting system.'
                           '(default: 13)')
    parser.add_argument('--gaussian_sigma', type=float, default=15,
                      help='Gaussian sigma for the reference images gaussian functions.'
                           '(default: 15)')
    parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
    parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
    return parser.parse_args()

def main():
    """Main function."""
    warnings.filterwarnings('ignore')
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    dataset_items = [item for item in sorted(os.listdir(args.dataset)) if (item != '.ipynb_checkpoints')]
    
    image_list_name = os.path.splitext(os.path.basename(dataset_items[0]))[0]    
    output_dir = args.output_dir or f'results/{image_list_name}'
    
    gerador = StyleGANGenerator(args.model_name)
    encoder = StyleGANEncoder(args.model_name)
    perceptual = PerceptualModel(min_val=gerador.min_val, max_val=gerador.max_val)
    
    discriminador = Discriminator(size=256).to(gerador.run_device)
    discriminador.load_state_dict(torch.load('models/pretrain/FFHQ256_940k_flip.pt')["d"])
    
    for dataset_item_idx in tqdm(range(len(dataset_items)), leave=False):    
        foldername = dataset_items[dataset_item_idx]
        
        image_list_name = os.path.splitext(os.path.basename(foldername))[0]            
        output_dir = os.path.join(f'results/', image_list_name)
        os.makedirs(output_dir, exist_ok=True)
        
        reference_images = [item for item in sorted(os.listdir(f'{args.dataset}/{foldername}/reference_imgs/')) if (item != '.ipynb_checkpoints')]
    
        print(f'Inverting reference images for {foldername}!')

        inverter = StyleGANInverter(
            args.model_name,
            learning_rate=args.learning_rate,
            iteration=args.num_iterations,
            reconstruction_loss_weight=1.0,
            perceptual_loss_weight=5e-5,
            regularization_loss_weight=2.0,
            mask = 0.0,
            latent_codes = 0.0,
            potential_function_loss = 0.0,
            discriminator_loss = 0.0,
            proportional_loss = 0.0,
            proportional_scale = 0.0,
            proportional_shift = 0.0,
            gaussian_sigma = 0.0,
            gerador = gerador,
            encoder = encoder,
            perceptual = perceptual,
            discriminador = discriminador)
        image_size = inverter.G.resolution

        # Initialize visualizer.
        save_interval = args.num_iterations // args.num_results
        headers = ['Name', 'Original Image', 'Encoder Output']
        for step in range(1, args.num_iterations + 1):
            if step == args.num_iterations or step % save_interval == 0:
                headers.append(f'Step {step:06d}')
        viz_size = None if args.viz_size == 0 else args.viz_size
        visualizer = HtmlPageVisualizer(
            num_rows=len(reference_images), num_cols=len(headers), viz_size=viz_size)
        visualizer.set_headers(headers)

        latent_codes = []
        for reference_img_idx in range(len(reference_images)):
            reference_img = reference_images[reference_img_idx]
            image_path = f'{args.dataset}/{foldername}/reference_imgs/{reference_img}'

            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image = resize_image(load_image(image_path), (image_size, image_size))
            return_invert, intersection = inverter.easy_invert(image, 0.0, num_viz=args.num_results)
            code = return_invert[0]
            viz_results = return_invert[1]
            latent_codes.append(code)
            visualizer.set_cell(reference_img_idx, 0, text=image_name)
            visualizer.set_cell(reference_img_idx, 1, image=image)
            for viz_idx, viz_img in enumerate(viz_results[1:]):
                visualizer.set_cell(reference_img_idx, viz_idx + 2, image=viz_img)

        # Save results.        
        np.save(f'{output_dir}/inverted_codes.npy', np.concatenate(latent_codes, axis=0))
        visualizer.save(f'{output_dir}/reference_images_inversion.html')

        print(f'Reconstucting occluded images for {foldername}!')

        occluded_files = [item for item in sorted(os.listdir(f'{args.dataset}/{foldername}/occluded_img/')) if (item != '.ipynb_checkpoints')]
        occlusion_files = [item for item in sorted(os.listdir(f'{args.dataset}/{foldername}/occlusion_mask/')) if (item != '.ipynb_checkpoints')]
        
        # Initialize visualizer.
        save_interval = args.num_iterations // args.num_results
        headers = ['Name', 'Original Image', 'Encoder Output']
        for step in range(1, args.num_iterations + 1):
            if step == args.num_iterations or step % save_interval == 0:
                headers.append(f'Step {step:06d}')
        viz_size = None if args.viz_size == 0 else args.viz_size
        visualizer = HtmlPageVisualizer(num_rows=len(occluded_files), num_cols=len(headers), viz_size=viz_size)
        visualizer.set_headers(headers)

        for occl_file_idx in range(len(occluded_files)):
            occluded_file = occluded_files[occl_file_idx]
            occlusion_mask = occlusion_files[occl_file_idx]

            #----------- Inverter a imagem ocluida já com os latent codes das reference images -----------
            latent_codes = np.load(f'{output_dir}/inverted_codes.npy')
            reference_img_count = np.shape(latent_codes)[0]

            # Initialize the inverter with the current hyperparameters combination
            inverter = StyleGANInverter(
                args.model_name,
                learning_rate=args.learning_rate,
                iteration=args.num_iterations,
                reconstruction_loss_weight=1.0,
                perceptual_loss_weight=args.loss_weight_feat,
                regularization_loss_weight=0.0,
                mask = 1.0,
                latent_codes = latent_codes,
                potential_function_loss = 0.0,
                discriminator_loss = args.loss_discriminator,
                proportional_loss = args.prior_flag,
                proportional_scale = 0.25,
                proportional_shift = args.proportional_shift,
                gaussian_sigma = args.gaussian_sigma,
                gerador = gerador,
                encoder = encoder,
                perceptual = perceptual,
                discriminador = discriminador)
            image_size = inverter.G.resolution

            # Load image list
            line = f'{args.dataset}/{foldername}/occluded_img/{occluded_file}'
            line1 = f'{args.dataset}/{foldername}/occlusion_mask/{occlusion_mask}'       
            parts = occluded_file.rsplit('_', 1)  # Split by the last underscore

            # Invert images.
            image_path = line.strip()
            mask_path = line1.strip()
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image = resize_image(load_image(image_path), (image_size, image_size))

            return_invert, intersection = inverter.easy_invert(image, mask_path, num_viz=args.num_results)
            code = return_invert[0]
            viz_results = return_invert[1]

            save_image(f'{output_dir}/{image_name}_ori.png', image)
            save_image(f'{output_dir}/{image_name}_enc.png', viz_results[1])
            save_image(f'{output_dir}/{image_name}_inv.png', viz_results[-1])
            visualizer.set_cell(occl_file_idx, 0, text=image_name)
            visualizer.set_cell(occl_file_idx, 1, image=image)
            for viz_idx, viz_img in enumerate(viz_results[1:]):
                visualizer.set_cell(occl_file_idx, viz_idx + 2, image=viz_img)   

        # Save results.
        np.save(f'{output_dir}/reconstuction_inverted_codes.npy', np.concatenate(latent_codes, axis=0))
        visualizer.save(f'{output_dir}/reconstruction.html')


if __name__ == '__main__':
    main()