import os
import torch
from torchvision.utils import save_image
from models.helper_functions import *
from models.networks import *
from preprocessing.image_dataset_cycle import *
from preprocessing.dataset import *
from models.sb_test import *
from inception import InceptionV3 as inception_v3
from options.test_options import test_parser
from utils.FID_dataset import *
from utils.FID_epoch import *
from utils.loss_criterions import *
from utils.KID_dataset import *
from utils.KID_epoch import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    args = test_parser.parse_args()

    num_threads = args.num_threads
    batch_size = args.batch_size
    serial_batches = args.serial_batches
    no_flip = args.no_flip
    aspect_ratio = args.aspect_ratio

    # create output images directory
    results_dir = os.makedirs(args.create_dir, exist_ok=True)
    
    sb_model_test = SBModel_test().to(device)
    
    pretrained_dict = torch.load("/kaggle/working/our_pretrained_sb_model.pth")
    model_dict = sb_model_test.state_dict()
    
    fid_list_test = []
    kid_list_test = []

    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    # Update the new model's dict with the pretrained dict
    model_dict.update(pretrained_dict)  
    sb_model_test.load_state_dict(model_dict)
    sb_model_test.eval()
    
    for i, (dataA, dataB) in enumerate(zip(test_dataloaderA, test_dataloaderB)):
        dataA = dataA.to(device)
        dataB = dataB.to(device)
        if i == 0:
            sb_model_test.data_dependent_initialize(dataA, dataB)
            sb_model_test.eval()
            
        # Unpack data from data loader
        sb_model_test.set_input(dataA, dataB)  
        # Run inference
        sb_model_test.forward()  
        
        fake_B_images = sb_model_test.Xt_1
        visualize_images(fake_B_images.to(device), title="Generated Zebras")
        
        # Save the generated images if needed
        save_image(fake_B_images, os.path.join(results_dir, f"generated_zebras_{i}.png"))
        
        #Compute FID 
        fretchet_dist = epoch_calculate_fretchet(dataB, fake_B_images.to(device), inception_v3)
        print('FID:', fretchet_dist)
        
        # Compute activations
        activations_real = epoch_calculate_activations(dataB, inception_v3, cuda=True)
        activations_fake = epoch_calculate_activations(fake_B_images.to(device), inception_v3, cuda=True)
        
        # Calculate KID
        kid_value = epoch_compute_mmd_simple(activations_real, activations_fake)
        print('KID:', kid_value)
        
        fid_list_test.append(fretchet_dist)
        kid_list_test.append(kid_value)
        