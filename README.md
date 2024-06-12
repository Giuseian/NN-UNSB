# Unpaired Image-to-Image translation via Neural Schrödinger Bridge
In this project a re-implementation of the methods proposed in the paper "[Unpaired Image-to-Image Translation via Neural Schrödinger Bridge](https://arxiv.org/abs/2305.15086)" is presented. 

## Project description 
The original paper introduces a novel approach to address the limitations of traditional diffusion models in unpaired image-to-image (I2I) translation tasks. Diffusion models, which are generative models simulating stochastic differential equations (SDEs), often rely on a Gaussian prior assumption, which may not be suitable for all types of data distributions.

The Schrödinger Bridge (SB) concept provides an effective method by training an SDE to transform between two distinct distributions. This approach incorporates sophisticated discriminators and regularization methods, enhancing the model's capability to efficiently translate between unpaired datasets.

The re-implementation aims to explore the scalability and efficiency of UNSB, demonstrating its capability to perform various unpaired I2I translation tasks, particularly focusing on high-resolution images where previous SB models have faced challenges.

## Dataset 
For our image translation experiments, we have selected the horse2zebra dataset. It's divided in 2 main domains: domain A for horses and domain B for zebras.
<p align="center">
  <img src="images/horse2zebra_dataset_composition.png" width="300">
</p>
Due to limited computational resources, we opted to train the model on a smaller subset of the original dataset. 
Specifically:

- **Training Datasets**: each of the horse and zebra categories in the training set contains 200 images, totaling 400 images for model training. 
- **Testing Datasets**: for testing, each category—horse and zebra—includes 120 images. 

The selection of these subsets was aimed at maintaining a diverse representation of images to ensure that the model can still learn to generalize well across different types of input. 

## Architecture
The model exhibits a complex structure tailored for generating high-quality images while ensuring that the generated content remains diverse and adheres closely to the real data distribution.

The goal is to start from an image of an horse and generate the same image but with features of a zebra. 

Unlike traditional GANs, it uses a form of the diffusion process, which involves gradually transforming a sample from a simple distribution into a complex data distribution over multiple time steps. 

Schrödinger Bridge is used to guide this diffusion process such that it not only generates images by removing noise but does so by following the most probable path under the constraints of starting with a noise distribution and ending with a data distribution. 

The model is made by several sub-networks including multiple discriminators and generators, which likely serve different purposes such as handling different aspects of the image data and ensuring diverse capabilities within the model.

Below there's a graphical representation of the reimplemented network structure:
<p align="center">
  <img src="images/structure_network.png">
</p>

## Repositiory Content
* **`extras`**: contains additional networks that were developed during the re-implementation of the paper so to experiment several techniques

* **`images`**: stores image files used in the project, such as generated images and evaluation plots

* **`models`**: 

* **`preprocessing`**: scripts dedicated to data cleaning, transformation, and preparation 

* **`utils`**: utility scripts to be used during evaluation stage


## Installation 

1. Clone the repository:
   ```bash
    git clone link_nostra_repo
    cd nome_cartella
    ```
2. Create and activate a virtual environment:
    ```bash
    python3 -m venv virtual 
    source virtual/bin/activate
    ```
TODO: in caso creare un file req.txt e mettere in  caso il comando per installarli nel ambiente 

## Training 

TODO: qui dobbiamo inserire il codice con il quale facciamo partire il training + commenti 

## Results 
 

<p align="center">
  <img src="images/zebra1_results.png" alt="Zebra 1" width="30%" />
  <img src="images/zebra2_results.png" alt="Zebra 2" width="30%" />
  <img src="images/zebra3_results.jpg" alt="Zebra 3" width="30%" />
</p>


<p align="center">
  <img src="images/FID_values_140_epochs.png" alt="FID" width="40%" />
  <img src="images/KID_values_140_epochs.png" alt="KID" width="40%" />
</p>

