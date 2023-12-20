# python 3.7
"""Utility functions to invert a given image back to a latent code."""

from tqdm import tqdm
import cv2
import numpy as np
import os
from sklearn.neighbors import KernelDensity

import torch
from torch.nn import functional as F
from torchvision import transforms

from models.stylegan_generator import StyleGANGenerator
from models.stylegan_encoder import StyleGANEncoder
from models.perceptual_model import PerceptualModel

#from models.styleGAN2_discriminator import Discriminator
from models.styleGAN2_model import Discriminator#, Generator

from utils.visualizer import save_image, load_image, resize_image
from utils.mask_polygon import get_maks_percentage

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
  (3) Regularization loss from encoder. (optional, but recommended for in-domain
      inversion)

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
               average_loss_weight=1.0,
               average_initial_loss_weight=0.0,
               mask=0.0,
               potential_function_loss=0.0,
               proportional_loss=0.0,
               logger=None):
    """Initializes the inverter.

    NOTE: Only Adam optimizer is supported in the optimization process.

    Args:
      model_name: Name of the model on which the inverted is based. The model
        should be first registered in `models/model_settings.py`.
      logger: Logger to record the log message.
      learning_rate: Learning rate for optimization. (default: 1e-2)
      iteration: Number of iterations for optimization. (default: 100)
      reconstruction_loss_weight: Weight for reconstruction loss. Should always
        be a positive number. (default: 1.0)
      perceptual_loss_weight: Weight for perceptual loss. 0 disables perceptual
        loss. (default: 5e-5)
      regularization_loss_weight: Weight for regularization loss from encoder.
        This is essential for in-domain inversion. However, this loss will
        automatically ignored if the generative model does not include a valid
        encoder. 0 disables regularization loss. (default: 2.0)
    """
    
    self.logger = logger
    self.model_name = model_name
    self.gan_type = 'stylegan'

    self.G = StyleGANGenerator(self.model_name, self.logger)
    #self.G1 = Generator(size=256, style_dim=512, n_mlp=8).to(self.G.run_device)
    #self.G1.load_state_dict(torch.load('models/pretrain/550000.pt')["g"], strict=False)
    self.E = StyleGANEncoder(self.model_name, self.logger)
    self.F = PerceptualModel(min_val=self.G.min_val, max_val=self.G.max_val)
    
    self.logger.info(f'Build network for module `Discriminator`.')
    self.D = Discriminator(size=self.G.resolution).to(self.G.run_device)
    self.logger.info(f'Loading pytorch weights from `models/pretrain/550000.pt`.')
    self.D.load_state_dict(torch.load('models/pretrain/550000.pt')["d"])
    self.logger.info(f'Successfully loaded!')   
    
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
    self.loss_ave_weight = average_loss_weight
    self.ave_init_loss_weight = average_initial_loss_weight
    self.mask = mask
    self.potential_function_loss = potential_function_loss
    self.proportional_loss = proportional_loss
    #assert self.loss_pix_weight > 0


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
    if self.ave_init_loss_weight:
        # Load the latent codes from the .npy file
        latent_codes = np.load('results/inversion/before_mask/sem_oclusao/300/inverted_codes.npy')

        #load the euclidean distance similarity matrix
        sim_matrix = np.load('analysis_output/300/sim_matrix.npy')

        #normalize the similarity matrix data between 0 and 1
        weights = (sim_matrix - np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))
        weights = 1 - weights

        #take the first row of that matrix excluding the element of the diagonal (currently inverting the first image)
        weights_row = weights[0, :]
        #weights_row = weights_row[1:]
        weights_row = weights_row[:]
        
        # calculate the average along axis 0
        #z = np.average(latent_codes[1:, :, :], axis=0, weights=weights_row)
        z = np.average(latent_codes[:, :, :], axis=0, weights=weights_row)
        # add a new axis at position 0
        z = np.expand_dims(z, axis=0)
        
    else:
        x = image[np.newaxis]
        x = self.G.to_tensor(x.astype(np.float32))
        z = _get_tensor_value(self.E.net(x).view(1, *self.encode_dim))
    
    #print(np.shape(z))
    return z.astype(np.float32)

  def invert(self, image, num_viz=0):
    """Inverts the given image to a latent code.

    Basically, this function is based on gradient descent algorithm.

    Args:
      image: Target image to invert, which is assumed to have already been
        preprocessed.
      num_viz: Number of intermediate outputs to visualize. (default: 0)

    Returns:
      A two-element tuple. First one is the inverted code. Second one is a list
        of intermediate results, where first image is the input image, second
        one is the reconstructed result from the initial latent code, remainings
        are from the optimization process every `self.iteration // num_viz`
        steps.
    """    
        
    x = image[np.newaxis]
    x = self.G.to_tensor(x.astype(np.float32))
    x.requires_grad = False
    init_z = self.get_init_code(image)
    z = torch.Tensor(init_z).to(self.run_device)
    z.requires_grad = True
    
    
    if self.mask:
        #mask = cv2.imread('analysis_output/300/mask.png', cv2.IMREAD_GRAYSCALE)
        #mask = cv2.imread('analysis_output/Al_gore/mask.png', cv2.IMREAD_GRAYSCALE)
        #mask = cv2.imread('analysis_output/Recep/mask_smallhand.png', cv2.IMREAD_GRAYSCALE)
        #mask = cv2.imread('analysis_output/Recep/mask_bighand.png', cv2.IMREAD_GRAYSCALE)
        #mask = cv2.imread('analysis_output/Abdullah_Gul/mask_bighand.png', cv2.IMREAD_GRAYSCALE)
        #mask = cv2.imread('analysis_output/face1030/mask.png', cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread('analysis_output/Recep/mask.png', cv2.IMREAD_GRAYSCALE)
        #mask = cv2.imread('analysis_output/face1030/mask_smallhand.png', cv2.IMREAD_GRAYSCALE)
        
        inv_mask = cv2.bitwise_not(mask)
        inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel mask for bitwise_and
        inv_mask = self.preprocess(inv_mask)
        inv_mask = inv_mask.reshape((1, 3, 256, 256))
        
        masked_x = torch.abs(x) * torch.abs(torch.torch.from_numpy(inv_mask).to(self.run_device))


    optimizer = torch.optim.Adam([z], lr=self.learning_rate)
    #optimizer = torch.optim.SGD([z], lr=self.learning_rate)

    viz_results_known_latent_codes = []
    viz_results = []
    viz_results.append(self.G.postprocess(_get_tensor_value(x))[0])
    x_init_inv = self.G.net.synthesis(z)
    viz_results.append(self.G.postprocess(_get_tensor_value(x_init_inv))[0])
    pbar = tqdm(range(1, self.iteration + 1), leave=True)
    
    if self.loss_ave_weight:
        # Load the latent codes from the .npy file (9, 14, 512) -> (n_imagens, n_iterations, dim)
        #latent_codes = np.load('results/inversion/before_mask/sem_oclusao/300/inverted_codes.npy')
        #latent_codes = np.load('results/inversion/Al_gore/inverted_codes.npy')
        #latent_codes = np.load('results/inversion/Al_gore_without6/inverted_codes.npy')
        #latent_codes = np.load('results/inversion/Abdullah_Gul/inverted_codes.npy')
        latent_codes = np.load('results/inversion/Abdullah_Gul_without5/inverted_codes.npy')

        #load the euclidean distance similarity matrix
        #sim_matrix = np.load('analysis_output/300/sim_matrix.npy')
        #sim_matrix = np.load('analysis_output/Al_gore/sim_matrix.npy')
        sim_matrix = np.load('analysis_output/Abdullah_Gul/sim_matrix.npy')

        #normalize the similarity matrix data between 0 and 1
        weights = (sim_matrix - np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))
        weights = 1 - weights

        #take the first row of that matrix excluding the element of the diagonal (currently inverting the first image)
        weights_row = weights[0, :]
        weights_row = weights_row[:]
        
        # calculate the average along axis 0
        z_ave = np.average(latent_codes[:, :, :], axis=0, weights=weights_row)
        z_ave = np.expand_dims(z_ave, axis=0)
        
    if (self.potential_function_loss or self.proportional_loss):
        # Load the latent codes from the .npy file
        #latent_codes = np.load('results/inversion/before_mask/sem_oclusao/300/inverted_codes.npy')
        #latent_codes = np.load('results/inversion/Al_gore/inverted_codes.npy')
        #latent_codes = np.load('results/inversion/Al_gore_without6/inverted_codes.npy')
        #latent_codes = np.load('results/inversion/Abdullah_Gul/inverted_codes.npy')
        #latent_codes = np.load('results/inversion/Abdullah_Gul_without5/inverted_codes.npy')
        #latent_codes = np.load('results/inversion/face1030/inverted_codes.npy')
        latent_codes = np.load('results/inversion/Recep/inverted_codes.npy')
        #latent_codes = np.load('results/inversion/Al_gore_true/inverted_codes.npy')
        #latent_codes = np.load('results/inversion/Al_gore-2faces/inverted_codes.npy')
        
        #stores the number of latent codes
        num_known_latents = np.shape(latent_codes)[0]

        """
        #load the euclidean distance similarity matrix
        #sim_matrix = np.load('analysis_output/300/sim_matrix.npy')
        sim_matrix = np.load('analysis_output/Al_gore/sim_matrix.npy')
        #sim_matrix = np.load('analysis_output/Abdullah_Gul/sim_matrix.npy')

        #normalize the similarity matrix data between 0 and 1
        weights = (sim_matrix - np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))
        weights = 1 - weights

        #take the first row of that matrix excluding the element of the diagonal (currently inverting the first image)
        weights_row = weights[0, :]
        #weights_row = weights_row[1:]
        weights_row = weights_row[:]
        """
    
    
    
    
    flag123 = True
    distances_final = []
    gaussians_final = []
    distances_true_known = []

    #image_path = 'my_examples/Al_gore/Al_Gore_0006_01.png'
    #image_path = 'my_examples/Abdullah_Gul/Abdullah_Gul_0005_01.png'
    #image_path = 'my_examples/00001.png'
    #image_path = 'my_examples/face1030/face1029_01.png'
    image_path = 'my_examples/Recep/Recep_Tayyip_Erdogan_0020_01.png'
    
    image_size = self.G.resolution
    true_image = resize_image(load_image(image_path), (image_size, image_size))
    true_image = self.preprocess(true_image)
    
    
    """
    # Generate a random matrix
    A = np.random.rand(512, 512)

    # Construct a symmetric matrix
    A = 0.5 * (A + A.T)

    # Add a small positive diagonal shift
    A += 512 * np.eye(512)

    # Ensure the matrix is positive semi-definite
    _, V = np.linalg.eigh(A)
    covariance_matrix = np.dot(V, np.dot(np.diag(np.abs(np.random.rand(512))), V.T))
    covariance_matrix = torch.from_numpy(covariance_matrix).float()
    covariance_matrix = 100 * covariance_matrix
    """
    
    for step in pbar:
      loss = 0.0

      #log_message = f'aaa'
    
      
      # Reconstruction loss.
      if (self.loss_pix_weight != 0 and self.mask):
            x_rec = self.G.net.synthesis(z)
            masked_xrec = torch.abs(x_rec) * torch.abs(torch.from_numpy(inv_mask).to(self.run_device))

            loss_pix = torch.mean((masked_x - masked_xrec) ** 2)
            loss = loss + loss_pix * self.loss_pix_weight
            log_message = f'loss_pix: {_get_tensor_value(loss_pix):.3f}'

      elif self.loss_pix_weight != 0:
          x_rec = self.G.net.synthesis(z)
          loss_pix = torch.mean((x - x_rec) ** 2)
          loss = loss + loss_pix * self.loss_pix_weight
          log_message = f'loss_pix: {_get_tensor_value(loss_pix):.3f}'
        
      else:
        x_rec = self.G.net.synthesis(z)
        log_message = f'aaa'

      # Perceptual loss.
      if (self.loss_feat_weight != 0 and self.mask):
        masked_xrec = torch.abs(x_rec) * torch.abs(torch.from_numpy(inv_mask).to(self.run_device))    
        x_feat = self.F.net(masked_x)        
        x_rec_feat = self.F.net(masked_xrec)
        loss_feat = torch.mean((x_feat - x_rec_feat) ** 2)
        loss = loss + loss_feat * self.loss_feat_weight
        log_message += f', loss_feat: {_get_tensor_value(loss_feat):.3f}'
        
      elif self.loss_feat_weight != 0:
        x_feat = self.F.net(x)
        x_rec_feat = self.F.net(x_rec)
        loss_feat = torch.mean((x_feat - x_rec_feat) ** 2)
        loss = loss + loss_feat * self.loss_feat_weight
        log_message += f', loss_feat: {_get_tensor_value(loss_feat):.3f}'

      # Regularization loss.
      if self.loss_reg_weight:
        z_rec = self.E.net(x_rec).view(1, *self.encode_dim)
        loss_reg = torch.mean((z - z_rec) ** 2)
        loss = loss + loss_reg * self.loss_reg_weight
        log_message += f', loss_reg: {_get_tensor_value(loss_reg):.3f}'
        
      # Average of latent codes from known images loss.
      if self.loss_ave_weight:
        loss_ave = torch.mean((z - torch.from_numpy(z_ave).to(self.run_device)) ** 2) #first move z_ave to the GPU using the to() method, and then subtract it from z
        loss = loss + loss_ave * self.loss_ave_weight
        log_message += f', loss_ave: {_get_tensor_value(loss_ave):.3f}'
        
        
      if self.potential_function_loss:
        #Define parameters for each gaussian
        sigma = 10
        A = 1
        gaussians = torch.tensor([], dtype=torch.float32).to(self.run_device)
        
        """
        if flag123:
            #known_latent_code = np.load('results/inversion/before_mask/sem_oclusao/300_1/inverted_codes.npy')
            #opt_latent_code = np.load('results/inversion/Abdullah_Gul_true/inverted_codes.npy')
            #opt_latent_code = np.load('results/inversion/Al_gore_true/inverted_codes.npy')
            opt_latent_code = np.load('results/inversion/face1030_true/inverted_codes.npy')

            distance_true_known = np.linalg.norm(opt_latent_code - latent_codes, axis=(1,2))
            distances_true_known.append(distance_true_known)
            
            flag123 = False
        """

        #Calculate the potential function for a given latent code and set of known latent codes
        for i in range(num_known_latents):
            diff = torch.from_numpy(latent_codes[i, :, :]).to(self.run_device) - z.squeeze()
            norm = torch.norm(diff, dim=1)
            gaussian = A * torch.exp(-norm / (2 * sigma**2))
            gaussians = torch.cat((gaussians, gaussian.unsqueeze(0)), dim=0)            

        distances = np.linalg.norm(z.detach().cpu().numpy() - latent_codes, axis=(1,2))
        distances_final.append(distances)
        
        gaussians_final.append(gaussians.detach().cpu().numpy())
        
        potential_value = torch.sum(gaussians)
        
        #true_distance = np.linalg.norm(opt_latent_code - z.detach().cpu().numpy())
        #distances = np.sqrt(np.sum(np.power(latent_codes[i, :, :] - z[:, :, :].detach().cpu().numpy(), 2)))
        
        
        x_to_compare = self.G.net.synthesis(z)
        reconstrucion_true = torch.mean((torch.from_numpy(true_image).to(self.run_device) - x_to_compare) ** 2)
        
        
        #log_message += f', true distance: {true_distance:.3f}'
        log_message += f', reconstruction distance: {reconstrucion_true:.3f}'
        #log_message += f', distances: {distances:.3f}'
        log_message += f', potential value: {potential_value:.3f}'
        
        potential_value = -1 * potential_value

        loss = loss + potential_value * self.potential_function_loss        

       
      if True:
        #Discriminator loss, we should maximize the output value as it represents a score of how realistic the input image is 
        x_rec = self.G.net.synthesis(z)
        
        """
        # Define the mean and standard deviation values used during training
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        # Create the transformation pipeline
        transform = transforms.Compose([
            #transforms.ToTensor(),  # Convert PIL Image to a PyTorch tensor (values between 0 and 1)
            transforms.Normalize(mean=mean, std=std)  # Normalize the tensor with the specified mean and std
        ])

        # Apply the transformation to 'image1'
        transformed_image = transform(x_rec)
        """

        d_result = self.D(x_rec)
        #d_result = self.D(torch.from_numpy(true_image).unsqueeze(0).to(self.run_device))
        d_loss = F.softplus(d_result).mean() #softplus is a smooth approximation to the ReLU

        log_message += f', discriminator_score: {d_result.mean():.3f}'
        log_message += f', discriminator_loss: {d_loss:.3f}'
        
        #d_result = -1 * d_result
        
        loss = loss + d_loss# * 10
        
        
        
      log_message += f', loss: {_get_tensor_value(loss):.3f}'
      pbar.set_description_str(log_message)
      if self.logger:
        self.logger.debug(f'Step: {step:05d}, '
                          f'lr: {self.learning_rate:.2e}, '
                          f'{log_message}')

      # Do optimization.
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if num_viz > 0 and step % (self.iteration // num_viz) == 0:
        viz_results.append(self.G.postprocess(_get_tensor_value(x_rec))[0])
        
      if step == 39:
        viz_results_known_latent_codes.append(self.G.postprocess(_get_tensor_value(torch.from_numpy(true_image.reshape((1, 3, 256, 256))).to(self.run_device)))[0])
        
    distances_true_known = np.array(distances_true_known)
    np.savetxt('distances_csv/distances_true_known.csv', distances_true_known, delimiter = ',')
    
    """
    sg2_image, _ = self.G1(z)
    
    print(np.shape(self.G.net.synthesis(z))) # (1, 3, 256, 256)
    print(np.shape(sg2_image)) # (14, 3, 256, 256)
    print(np.shape(sg2_image[0, :, :, :])) # (3, 256, 256)
    print(np.shape(z)) # (1, 14, 512)    
    print(np.shape(sg2_image[0, :, :, :].permute(1, 2, 0))) #(256, 256, 3)
    
    
    viz_results_known_latent_codes.append(sg2_image[5, :, :, :].permute(1, 2, 0).cpu().detach().numpy())#.transpose(1, 2, 0))"""
    
    distances_final = np.array(distances_final)
    #gaussians_final = np.array(gaussians_final)
    np.savetxt('distances_csv/distances.csv', distances_final, delimiter=',')
    #np.savetxt('distances_csv/gaussians_1_100.csv', gaussians_final, delimiter=',')
    
    #np.save('distances_csv/distances_full.npy', distances)
    #np.save('distances_csv/z.npy', z[:, -1, :].detach().cpu().numpy())
    
    """
    viz_results_known_latent_codes = []
    latent_image = self.G.net.synthesis(torch.from_numpy(np.expand_dims(latent_codes[i, :, :], axis=0)).to(self.run_device))
    viz_results_known_latent_codes.append(self.G.postprocess(_get_tensor_value(latent_image))[0])
    """
    return _get_tensor_value(z), viz_results, viz_results_known_latent_codes





  def easy_invert(self, image, num_viz=0):
    """Wraps functions `preprocess()` and `invert()` together."""
    
    if self.proportional_loss and self.mask:
        #mask = cv2.imread('analysis_output/Al_gore/mask.png', cv2.IMREAD_GRAYSCALE)
        #mask = cv2.imread('analysis_output/Recep/mask_smallhand.png', cv2.IMREAD_GRAYSCALE)
        #mask = cv2.imread('analysis_output/Recep/mask_bighand.png', cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread('analysis_output/Recep/mask.png', cv2.IMREAD_GRAYSCALE)
        #mask = cv2.imread('analysis_output/Abdullah_Gul/mask_bighand.png', cv2.IMREAD_GRAYSCALE)
        #mask = cv2.imread('analysis_output/face1030/mask_smallhand.png', cv2.IMREAD_GRAYSCALE)
        
        mask_percentage, intersection = get_maks_percentage(image, mask, self.G.resolution, save_image_flag=True)
        
        if mask_percentage < 10:
            percentage_to_loss = 0
        else:
            percentage_to_loss = (1/25)*mask_percentage
            
        self.potential_function_loss = percentage_to_loss
    else:
        intersection = -1
    
    return self.invert(self.preprocess(image), num_viz), intersection