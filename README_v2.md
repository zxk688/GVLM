# GVLM v1.1 

# 1. Introduction

**GVLM v1.1** is a multi-modal version of GVLM. It contains geo-referenced remote sensing images with high-quality labels.

# 2. Event Details
This dataset covers 7 typical landslide events in recent years. Sites are standardized as "City/Region, Country" with their dates following the YYYY-MM-DD format.


| ID | Site (City/Region, Country) | Image Size (Pixels) | Trigger | Event Date |
| --- | --- | --- | --- | --- | 
| **1** | Mtauchira, Malawi | 10496 x 12288 | Cyclone | 2023-03 |
| **2** | Wuxie, China | 20480 x 17920 | Rainstorm | 2021-06-10 | 
| **3** | Longchuan, China | 18176 x 15360 | Rainstorm | 2019-06-13 |
| **4** | Te Haroto, New Zealand | 20480 x 22528 | Cyclone | 2023-02-21 | 
| **5** | São Sebastião, Brazil | 20736 x 11520 | Urban Growth | 2023-02 |
| **6** | Nippes, Haiti | 21504 x 12288 | Earthquake | 2021-08-14 |
| **7** | Enshi, China | 13824 x 10496 | Rainfall | 2021-06-21 | 


# 3. Technical Specifications

Each event provides multi-temporal imagery: Pre-event (`img1`) and Post-event (`img2`).
* **High-Resolution Optical (HR-RGB)**
* **Resolution**: ~1.2 m/pixel
* **Bands**: 3 (Red, Green, Blue) | `uint8`


* **Multi-Spectral (MS)**
* **Resolution**: ~10 m/pixel
* **Bands**: 4 (B2: Blue, B3: Green, B4: Red, B8: NIR) | `uint8`


* **Synthetic Aperture Radar (SAR)**
* **Resolution**: ~10 m/pixel
* **Polarization**: Dual-pol (VV, VH) | `int8`



# 4. Directory Structure

```text
GVLM_1.1_8bit/
├── [Site_Name]/
    ├── img1_HR.tif   # Pre-event High-Resolution RGB
    ├── img1_MS.tif   # Pre-event Multi-Spectral
    ├── img1_SAR.tif  # Pre-event SAR
    ├── img2_HR.tif   # Post-event High-Resolution RGB
    ├── img2_MS.tif   # Post-event Multi-Spectral
    ├── img2_SAR.tif  # Post-event SAR
    └── label.tif     # Landslide Ground Truth (Binary Mask)

```
# 5. Download

GVLM_1.1_8bit.zip from Baidu Netdisk or Google Drive:
* **Baidu Netdisk**: [Access Link](https://pan.baidu.com/s/1uLgchmxkToGSAJ3iAOQsdw?pwd=psnv) (PWD: `psnv`)
* **Google Drive**: [Access Link](https://drive.google.com/file/d/1DzeMZQuTVC5xJF00B0i1Jv8xXYXwIX7P/view?usp=sharing)


# 6. Acknowledgement

We would like to thank the [Google Earth platform](https://earth.google.com/) and [ESA Sentinel missions](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/The_Sentinel_missions)  for providing the remote sensing images.







