# Non-uniform Motion Blur Kernel Estimation via Adaptive Decomposition
Official Pytorch Implementation  of Non-uniform Motion Blur Kernel Estimation via Adaptive Decomposition [<a href="https://arxiv.org/abs/2102.01026">ArXiv</a>]
<p align="center">
<img width="700" src="imgs/Comparison_kernels.png?raw=true">
</p>

## Network Architecture
<p align="center">
<img width="900" src="imgs/architecture.png?raw=true">
  </p>
  
## Getting Started

### Clone Repository
```
git clone https://github.com/GuillermoCarbajal/NonUniformBlurKernelEstimationViaAdaptiveBasisDecomposition
```

### Download the pretrained model

Model can be downloaded from [here](https://www.dropbox.com/s/ro9smg1i7lh5b8d/TwoHeads.pkl?dl=0)
### Compute kernels from an image
```
python compute_kernels.py -i image_path -m model_path
```


### Deblur an image or a list of images
```
python image_deblurring.py -b blurry_img_path --reblur_model model_path --output_folder results
```

### Parameters
Additional options:   
  `--blurry_images`: may be a singe image path or a .txt with a list of images.
  
  `--n_iters`: number of iterations in the RL optimization (default 30)       
  
  `--resize_factor`: input image resize factor (default 1)     
  
  `--saturation_method`: `'combined'` or `'basic'`. When `'combined'` is passed RL in the presence of saturated pixels is applied. Otherwise,  simple RL update rule is applied in each iteration. For Kohler images, `'basic'` is applied. For RealBlur images `'combined'` is better.
  
  `--gamma_factor`: gamma correction factor. By default is assummed `gamma_factor=2.2`. For Kohler dataset images `gamma_factor=1.0`.
  

    
## Citation
If you use this code for your research, please cite our paper <a href="https://arxiv.org/abs/2102.01026"> Non-uniform Motion Blur Kernel Estimation via Adaptive Decomposition</a>:

```
@article{carbajal2021single,
  title={Non-uniform Motion Blur Kernel Estimation via Adaptive Decomposition},
  author={Carbajal, Guillermo and Vitoria, Patricia and Delbracio, Mauricio and Mus{\'e}, Pablo and Lezama, Jos{\'e}},
  journal={arXiv e-prints},
  pages={arXiv--2102},
  year={2021}
}
```
## Aknowledgments 

GC was supported partially by Agencia Nacional de Investigacion e Innovación (ANII, Uruguay) ´grant POS FCE 2018 1 1007783 and PV by the MICINN/FEDER UE project under Grant PGC2018- 098625-B-I0; H2020-MSCA-RISE-2017 under Grant 777826 NoMADS and Spanish Ministry of Economy and Competitiveness under the Maria de Maeztu Units of Excellence Programme (MDM-2015-0502). The experiments presented in this paper were carried out using ClusterUY (site: https://cluster.uy) and GPUs donated by NVIDIA Corporation. We also thanks Juan F. Montesinos for his help during the experimental phase.
