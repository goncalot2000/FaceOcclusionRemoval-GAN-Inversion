# python 3.7
"""Utility functions to invert a given image back to a latent code."""

from tqdm import tqdm
import cv2
import numpy as np
import os

import torch
from torch.nn import functional as F
from torchvision import transforms

from utils.visualizer import save_image, load_image, resize_image
from utils.mask_polygon import get_mask_percentage

__all__ = ['StyleGANInverter']


def _softplus(x):
  """Implements the softplus function."""
  return torch.nn.functional.softplus(x, beta=1, threshold=10000)

def _get_tensor_value(tensor):
  """Gets the value of a torch Tensor."""
  return tensor.cpu().detach().numpy()


class StyleGANInverter(object):
  """Defines the class for StyleGAN inversion.

  Even having the encoder, the output latent code is not good enough to recover
  the target image satisfyingly. To this end, this class optimize the latent
  code based on gradient descent algorithm. In the optimization process,
  following loss functions will be considered:

  (1) Pixel-wise reconstruction loss. (required)
  (2) Perceptual loss. (optional, but recommended)
  (3) Regularization loss from encoder. (optional)
  (4) Potential function loss (required when using prior knowledge)
  (5) Discriminator loss (optional)

  NOTE: The encoder can be missing for inversion, in which case the latent code
  will be randomly initialized and the regularization loss will be ignored.
  """

  def __init__(self,
               model_name,
               learning_rate=1e-2,
               iteration=100,
               reconstruction_loss_weight=1.0,
               perceptual_loss_weight=5e-5,
               regularization_loss_weight=2.0,
               mask=0.0,
               latent_codes = 0,
               potential_function_loss=0.0,
               discriminator_loss=0.0,
               proportional_loss=0.0,
               proportional_scale=0.25,
               proportional_shift=0.0,
               prior_flag = 0.0,
               gaussian_sigma = 10.0,
               gerador = 0.0,
               encoder = 0.0,
               perceptual = 0.0,
               discriminador = 0.0,
               logger=None):
    """Initializes the inverter.

    NOTE: Only Adam optimizer is supported in the optimization process."""
    
    self.logger = logger
    self.model_name = model_name
    self.gan_type = 'stylegan'

    self.G = gerador
    self.E = encoder
    self.F = perceptual

    self.D = discriminador   
    
    self.encode_dim = [self.G.num_layers, self.G.w_space_dim]
    self.run_device = self.G.run_device
    assert list(self.encode_dim) == list(self.E.encode_dim)

    assert self.G.gan_type == self.gan_type
    assert self.E.gan_type == self.gan_type

    self.learning_rate = learning_rate
    self.iteration = iteration
    self.loss_pix_weight = reconstruction_loss_weight
    self.loss_feat_weight = perceptual_loss_weight
    self.loss_reg_weight = regularization_loss_weight
    self.mask_flag = mask
    self.mask = 0
    self.potential_function_loss = potential_function_loss
    self.proportional_loss = proportional_loss
    self.proportional_scale = proportional_scale
    self.proportional_shift = proportional_shift
    self.prior_flag = prior_flag
    self.discriminator_loss = discriminator_loss
    self.gaussian_sigma = gaussian_sigma
    
    self.latent_codes = latent_codes


  def preprocess(self, image):
    """Preprocesses a single image.

    This function assumes the input numpy array is with shape [height, width,
    channel], channel order `RGB`, and pixel range [0, 255].

    The returned image is with shape [channel, new_height, new_width], where
    `new_height` and `new_width` are specified by the given generative model.
    The channel order of returned image is also specified by the generative
    model. The pixel range is shifted to [min_val, max_val], where `min_val` and
    `max_val` are also specified by the generative model.
    """
    if not isinstance(image, np.ndarray):
      raise ValueError(f'Input image should be with type `numpy.ndarray`!')
    if image.dtype != np.uint8:
      raise ValueError(f'Input image should be with dtype `numpy.uint8`!')

    if image.ndim != 3 or image.shape[2] not in [1, 3]:
      raise ValueError(f'Input should be with shape [height, width, channel], '
                       f'where channel equals to 1 or 3!\n'
                       f'But {image.shape} is received!')
    if image.shape[2] == 1 and self.G.image_channels == 3:
      image = np.tile(image, (1, 1, 3))
    if image.shape[2] != self.G.image_channels:
      raise ValueError(f'Number of channels of input image, which is '
                       f'{image.shape[2]}, is not supported by the current '
                       f'inverter, which requires {self.G.image_channels} '
                       f'channels!')

    if self.G.image_channels == 3 and self.G.channel_order == 'BGR':
      image = image[:, :, ::-1]
    if image.shape[1:3] != [self.G.resolution, self.G.resolution]:
      image = cv2.resize(image, (self.G.resolution, self.G.resolution))
    image = image.astype(np.float32)
    image = image / 255.0 * (self.G.max_val - self.G.min_val) + self.G.min_val
    image = image.astype(np.float32).transpose(2, 0, 1)

    return image

  def get_init_code(self, image):
    """Gets initial latent codes as the start point for optimization.

    The input image is assumed to have already been preprocessed, meaning to
    have shape [self.G.image_channels, self.G.resolution, self.G.resolution],
    channel order `self.G.channel_order`, and pixel range [self.G.min_val,
    self.G.max_val].
    """

    x = image[np.newaxis]
    x = self.G.to_tensor(x.astype(np.float32))
    z = _get_tensor_value(self.E.net(x).view(1, *self.encode_dim))
    
    return z.astype(np.float32)

  def invert(self, image, num_viz=0):
    """Inverts the given image to a latent code.

    Basically, this function is based on gradient descent algorithm.

    Args:
      image: Target image to invert, which is assumed to have already been
        preprocessed.
      num_viz: Number of intermediate outputs to visualize. (default: 0)

    Returns:
      A three-element tuple. First one is the inverted code. Second one is a list
        of intermediate results, where first image is the input image, second
        one is the reconstructed result from the initial latent code, remainings
        are from the optimization process every `self.iteration // num_viz`
        steps and third one is the loss value obtained.
    """    
        
    x = image[np.newaxis]
    x = self.G.to_tensor(x.astype(np.float32))
    x.requires_grad = False
    init_z = self.get_init_code(image)
    z = torch.Tensor(init_z).to(self.run_device)
    z.requires_grad = True
    
    if self.mask_flag:
        
        inv_mask = cv2.bitwise_not(self.mask)     

        inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel mask for bitwise_and
        inv_mask = self.preprocess(inv_mask)
        inv_mask = inv_mask.reshape((1, 3, 256, 256))
        
        masked_x = torch.abs(x) * torch.abs(torch.torch.from_numpy(inv_mask).to(self.run_device))


    optimizer = torch.optim.Adam([z], lr=self.learning_rate)

    viz_results = []
    viz_results.append(self.G.postprocess(_get_tensor_value(x))[0])
    x_init_inv = self.G.net.synthesis(z)
    viz_results.append(self.G.postprocess(_get_tensor_value(x_init_inv))[0])
    pbar = range(1, self.iteration + 1)
    
        
    if (self.potential_function_loss or self.proportional_loss):
        # Load the latent codes from the .npy file
        latent_codes = self.latent_codes
        
        # Stores the number of latent codes
        num_known_latents = np.shape(latent_codes)[0]

    
    for step in pbar:
      loss = 0.0
    
      # Reconstruction loss.
      if (self.loss_pix_weight != 0 and self.mask_flag):
            x_rec = self.G.net.synthesis(z)
            masked_xrec = torch.abs(x_rec) * torch.abs(torch.from_numpy(inv_mask).to(self.run_device))

            loss_pix = torch.mean((masked_x - masked_xrec) ** 2)
            loss = loss + loss_pix * self.loss_pix_weight

      elif self.loss_pix_weight != 0:
          x_rec = self.G.net.synthesis(z)
          loss_pix = torch.mean((x - x_rec) ** 2)
          loss = loss + loss_pix * self.loss_pix_weight
        
      else:
        x_rec = self.G.net.synthesis(z)

      # Perceptual loss.
      if (self.loss_feat_weight != 0 and self.mask_flag):
        masked_xrec = torch.abs(x_rec) * torch.abs(torch.from_numpy(inv_mask).to(self.run_device))    
        x_feat = self.F.net(masked_x)        
        x_rec_feat = self.F.net(masked_xrec)
        loss_feat = torch.mean((x_feat - x_rec_feat) ** 2)
        loss = loss + loss_feat * self.loss_feat_weight
        
      elif self.loss_feat_weight != 0:
        x_feat = self.F.net(x)
        x_rec_feat = self.F.net(x_rec)
        loss_feat = torch.mean((x_feat - x_rec_feat) ** 2)
        loss = loss + loss_feat * self.loss_feat_weight

      # Regularization loss.
      if self.loss_reg_weight:
        z_rec = self.E.net(x_rec).view(1, *self.encode_dim)
        loss_reg = torch.mean((z - z_rec) ** 2)
        loss = loss + loss_reg * self.loss_reg_weight      
        
      if self.potential_function_loss:
        #Define parameters for each gaussian
        A = 1
        gaussians = torch.tensor([], dtype=torch.float32).to(self.run_device)

        #Calculate the potential function for a given latent code (z) and set of known latent codes
        for i in range(num_known_latents):
            diff = torch.from_numpy(latent_codes[i, :, :]).to(self.run_device) - z.squeeze()
            norm = torch.norm(diff, dim=1)
            gaussian = A * torch.exp(-norm / (2 * self.gaussian_sigma**2))
            gaussians = torch.cat((gaussians, gaussian.unsqueeze(0)), dim=0)            
        
        potential_value = torch.sum(gaussians)        
        potential_value = -1 * potential_value

        loss = loss + potential_value * self.potential_function_loss        

       
      if self.discriminator_loss:
        x_rec = self.G.net.synthesis(z)

        d_result = self.D(x_rec)
        d_loss = F.softplus(d_result).mean()
        
        loss = loss + d_loss * self.discriminator_loss

      # Do optimization.
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if num_viz > 0 and step % (self.iteration // num_viz) == 0:
        viz_results.append(self.G.postprocess(_get_tensor_value(x_rec))[0])
        
    return _get_tensor_value(z), viz_results, loss


  def easy_invert(self, image, mask_path, num_viz=0):
    """Receives the binary occlusion mask and with its dimensions calculates the weights for the reconstruction loss and the gaussian loss.
    Wraps functions `preprocess()` and `invert()` together."""
    
    if self.mask_flag:
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if self.proportional_loss and self.mask_flag:
        mask_percentage, intersection = get_mask_percentage(image, self.mask, self.G.resolution, save_image_flag=False)
            
        # A sigmoidal function: scale controls how steep the transition is, and shift controls the midpoint of the transition.
        weight = 2 * (1 / (1 + np.exp(-self.proportional_scale * (mask_percentage - self.proportional_shift))))
        self.potential_function_loss = weight
        
    else:
        intersection = -1
    
    return self.invert(self.preprocess(image), num_viz), intersection