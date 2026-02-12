# spa-former
Code for the paper titled "Sparse Self-Attention Transformer for Image Inpainting".


<br>
This is the code for Sparse Self-Attention Transformer (Spa-former) to reconstruct corrupted image. Given one image and mask, the proposed **Spa-former** model is able to reconstruct masked regions. This code is adapted from an initial fork of [PIC](https://github.com/lyndonzheng/Pluralistic-Inpainting) implementation.

## Illustration of Spa-former
![](https://github.com/huangwenwenlili/spa-former/blob/main/images/spa-former-architecture.png)

Learning-based image inpainting methods have made remarkable progress in recent years. Nevertheless, these methods still suffer from issues such as blurring, artifacts, and inconsistent contents. The use of vanilla convolution kernels, which have limited perceptual fields and spatially invariant kernel coefficients, is one of the main causes for these problems. In contrast, the multi-headed attention in the transformer can effectively model non-local relations among input features by generating adaptive attention scores. Therefore, this paper explores the feasibility of employing the transformer model for the image inpainting task. However, the multi-headed attention transformer blocks pose a significant challenge due to their overwhelming computational cost. To address this issue, we propose a novel U-Net style transformer-based network for the inpainting task, called the sparse self-attention transformer (Spa-former). The Spa-former retains the long-range modeling capacity of transformer blocks while reducing the computational burden. It incorporates a new channel attention approximation algorithm that reduces attention calculation to linear complexity. Additionally, it replaces the canonical softmax function with the ReLU function to generate a sparse attention map that effectively excludes irrelevant features. As a result, the Spa-former achieves effective long-range feature modeling with fewer parameters and lower computational resources. Our empirical results on challenging benchmarks demonstrate the superior performance of our proposed Spa-former over state-of-the-art approaches.



# Getting started
## Installation
This code was tested with Pytoch 1.8.1 CUDA 11.1, Python 3.6 and Ubuntu 18.04

- Create conda environment:

```
conda create -n inpainting-py36 python=3.6
conda deactivate
conda activate inpainting-py36
pip install visdom dominate
```
- Clone this repo:

```
git clone https://github.com/huangwenwenlili/spa-former
cd spa-former
```

- Pip install libs:

```
pip install -r requirements.txt
```

## Datasets
- ```Paris StreetView```: It contains buildings of Paris of natural digital images. 14900 training images and 100 testing images. [Paris](https://github.com/pathak22/context-encoder)
- ```CelebA-HQ```: It contains celebrity face images. 30000 images. [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ)
- ```Places365-Standard```: It is the major part of the places2 dataset and was released by MIT. It has over 1.8 million training images and about 32K test images from 365 scene categories. We selected 1,000 pictures from the test set randomly to test the model. 

## Train
- Train the model. Input images and masks resolution are 256*256. We produce random irregular mask to corrupt images for training stage.
```
python train.py --name paris --checkpoints_dir ./checkpoints/checkpoint_paris --img_file /home/hwl/hwl/datasets/paris/paris_train_original/ --niter 261000 --batchSize 4 --lr 1e-4 --gpu_ids 0 --no_augment --no_flip --no_rotation 
```
- Set ```--mask_type``` in options/base_options.py to test various masks. ```--mask_file``` path is needed for **2 and 4 . random irregular mask**.
- ```--lr``` is learn rate, train scratch is 1e-4, finetune is 1e-5.

## Testing

- Test the model. Input images and masks resolution are 256*256. In the testing, we use [irregular mask dataset](https://github.com/NVIDIA/partialconv) to evaluate different ratios of corrupted region images.

```
python test.py  --name paris --checkpoints_dir ./checkpoints/checkpoint_paris --gpu_ids 0 --img_file your_image_path --mask_file your_mask_path --batchSize 1 --results_dir your_image_result_path
```
- Set ```--mask_type``` in options/base_options.py to test various masks. ```--mask_file``` path is needed for **3. external irregular mask**,
- The default results will be saved under the *results* folder. Set ```--results_dir``` for a new path to save the result.
- checkpooints [Baidu disk](https://pan.baidu.com/s/1Ace1zD_lUg-_KW7v_aVAOg?pwd=neg1); [Google Drive](https://drive.google.com/drive/folders/1vg6RoavdoZI8KeCeFVRVgcSg5Va4rB2P?usp=sharing)


## Example Results
- **Comparison results of softmax-based attention and our proposed Spa-attention**
![](https://github.com/huangwenwenlili/spa-former/blob/main/images/spa-intr.png)

(a) Attention values were computed for one feature channel using the softmax and ReLU functions. We found that the ReLU function generated attention values that were more focused on essential contexts, compared to the dense attention values obtained from the softmax function. 

(b) We compared the inpainting results obtained using these two attention mechanisms. Our Spa-attention approach yielded superior results, as indicated by the improved completion of the building window in the Spa-attention completion image, and the smaller FID value obtained by this method. Lower values of FID are indicative of better performance

## License
<br />
The codes and the pre-trained models in this repository are under the MIT license as specificed by the LICENSE file.
This code is for educational and academic research purpose only.

## Reference Codes
- https://github.com/lyndonzheng/Pluralistic-Inpainting
- https://github.com/NVIDIA/partialconv

## Citation

If you use this code for your research, please cite our paper.
```
@article{huang2023spaformer,
  title={Sparse Self-Attention Transformer for Image Inpainting},
  author={Huang, Wenli and Deng, Ye and Hui, Siqi and Zhou, Sanping and Wang, Jinjun},
  journal={},
  volume={},
  pages={},
  year={}
}
```
