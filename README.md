
# GVLM Dataset Version 1.0

The Global Very-High-Resolution Landslde Mapping (GVLM) dataset is the first large-scale and open-source VHR landslide mapping dataset. It is available for free to researchers for **only non-commercial use**. 

# Description
It includes $17$ bitemporal very-high-resolution imagery pairs with a spatial resolution of $0.59$ m acquired via Google Earth service. Each sub-dataset contains a pair of bitemporal images and the corresponding ground-truth map. The total coverage of the dataset is $163.77 km2$. The landslide sites in different geographical locations have various sizes, shapes, occurrence times, spatial distributions, phenology states, and land cover types, resulting in considerable spectral heterogeneity and intensity variations in the remote sensing imagery. The GVLM dataset can be used to develop and evaluate machine/deep learning models for change detection, semantic segmentation and landslide extraction. For more details, please refer to [Cross-domain landslide mapping from large-scale remote sensing images using prototype-guided domain-aware progressive representation learning](https://www.sciencedirect.com/science/article/abs/pii/S0924271623000242?dgcid=author).
![Locations](https://github.com/ZXK-RS/GVLM/blob/main/locnew.png)
If you find this dataset useful for your research, please cite our paper:
```
@article{ZHANG20231,
title = {Cross-domain landslide mapping from large-scale remote sensing images using prototype-guided domain-aware progressive representation learning},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {197},
pages = {1-17},
year = {2023},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2023.01.018},
url = {https://www.sciencedirect.com/science/article/pii/S0924271623000242},
author = {Xiaokang Zhang and Weikang Yu and Man-On Pun and Wenzhong Shi},
}
```
# Download
It is avaiable at  [Baidu Drive](https://pan.baidu.com/s/1OiP_LdpfPa4BBZqHAVeLXw)(extraction code:rkrg), or [Google Drive](https://drive.google.com/file/d/1RW8looAdRrZ6hanUWHaGCnmaHCkQZevP/view?usp=share_link).
In the future, we will continue promoting the establishment of a worldwide landslide mapping system by acquiring more remote sensing image data in landslide-prone areas. Hope you can join us!

# Demo
We also provide a simple demo for image clipping, model training and testing. Please refer to *LandslideMappingDemo*.
Users can split images into desired-size patches and generate their own train, validation, and test sets.

# Contact
Dr. Xiaokang Zhang (natezhangxk@gmail.com)

# Acknowledgement
We would like to thank [Google Earth platform](https://earth.google.com/) for providing the remote sensing images.
