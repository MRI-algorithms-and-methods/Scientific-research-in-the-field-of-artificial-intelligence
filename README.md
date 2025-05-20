![Made with ❤️](https://img.shields.io/badge/AI%20-%E2%9D%A4-golden)
[![Made with ❤️](https://img.shields.io/badge/ITMO%20-%E2%9D%A4-red)](https://itmo.ru/)
[![Made with ❤️](https://img.shields.io/badge/Our%20group%20-%E2%9D%A4-blue)](https://physics.itmo.ru/en/research-group/5202)
[![Dataset](https://img.shields.io/badge/Data-Available-brightgreen.svg)](https://drive.google.com/file/d/1cIJ-pzP3Sd16GqXRqOqlJ0tyiZ0slTiW/view?usp=sharing)


# Generation of realistic physics-based synthetic magnetic resonance imaging data

![image](https://github.com/user-attachments/assets/84871bd0-548e-4f7b-8ee6-e6345b08c95b)

# Overview

The project focuses on developing a platform for generating realistic synthetic MRI data, which can be used for various applied tasks. 

The platform consists of several blocks: 

- **Data Collection** — Collects real MRI data with annotations, tissue physical parameters, and information about magnetic field inhomogeneities, noise, and artifacts.
  
- **[Digital Phantom Creation](https://github.com/MRI-algorithms-and-methods/Scientific-research-in-the-field-of-artificial-intelligence/tree/main/MRI_phantom)** — Using the collected data, realiatic digital phantoms are created. We developed  and AI models ( **[diffusion-based](https://github.com/MRI-algorithms-and-methods/Scientific-research-in-the-field-of-artificial-intelligence/tree/main/Fast-DDPM_for_phantoms)**  , GAN-based) for generation realistic digital MR phantoms.
  
- **[MRI Data Synthesis (pipeline)](https://github.com/MRI-algorithms-and-methods/Scientific-research-in-the-field-of-artificial-intelligence/tree/main/pipeline)** — Synthesizing additional MRI data using an MRI scanner simulator and pulse sequences developed by the team. For  an example of such synthetic data refere to **[Generated_dataset](https://github.com/MRI-algorithms-and-methods/Scientific-research-in-the-field-of-artificial-intelligence/tree/main/Generated_dataset)** used for training **[DL-based  Bloch simulator](https://github.com/MRI-algorithms-and-methods/Scientific-research-in-the-field-of-artificial-intelligence/tree/main/DL-based_Bloch_simulator)**
  
- **Validation** — Testing the platform on applied tasks, such as:
    - **[Enhancing of segmentation quality of  MR image by synthetic data](https://github.com/MRI-algorithms-and-methods/Scientific-research-in-the-field-of-artificial-intelligence/tree/main/brain_segmentation_lib)**.
    -  MR image reconstruction for ultra-low-field MRI using deep learning algorithms (upcoming)

The project involves collaboration with NIITFA and JET LAB LLC, as well as close interaction with the international scientific community.

-----


## DL-based generation of realistic digital MRI brain phantom 
We investigated several DL-models for generation of realistic digital MRI brain phantom. Below comparsion of 2 models:
Fast Denoising Diffusion Probabulistic Model (**[FastDDPM](https://github.com/MRI-algorithms-and-methods/Scientific-research-in-the-field-of-artificial-intelligence/tree/main/Fast-DDPM_for_phantoms)** and conditional GAN)

| Method      | Map Type | Mean SSIM ± SD       | PSNR (dB) ± SD          |
|-------------|----------|-----------------------|------------------------|
| **FastDDPM**| T1       | 0.78 ± 0.05           | 22.12 ± 0.55           |
|             | T2       | 0.79 ± 0.02           | 18.67 ± 0.45           |
|             | PD       | 0.55 ± 0.04           | 19.32 ± 0.35           |
|   **CGAN**  | T1       | 0.72 ± 0.02           | 19.42 ± 0.45           |
|             | T2       | _ _ ±  _ _            | _ _ ±  _ _             |
|             | PD       | _ _ ±  _ _            | _ _ ±  _ _             |

## Dataset
This repository contains a **synthetic MRI dataset** with **23,329 samples** across **T1, T2, and PD contrasts**, split into:

- **Train**: 16,867 samples  
- **Test**: 6,462 samples

> ⚠️ **Note**: Due to GitHub file size restrictions, only a few **example files** are included.  
> The full dataset is available via the link below.


**[Learn more about dataset](https://github.com/MRI-algorithms-and-methods/Scientific-research-in-the-field-of-artificial-intelligence/tree/main/Generated_dataset)**

## Disclaimer

It is expressly understood that **ITMO University** does not make any warranties regarding the synthetic MRI data provided. Specifically, ITMO University does **not warrant or guarantee** that:

- The data is accurate;
- The data is merchantable;
- The data is fit for any particular purpose, including but not limited to clinical, diagnostic, or commercial applications.

**Use of this data is entirely at your own risk.**

ITMO University **shall not be held liable** for any claims, damages, or losses incurred by you or any third party as a result of using the data, whether directly or indirectly.

By accessing or using the synthetic MRI data, you acknowledge and accept these terms.

---

### License

This dataset is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license. You are free to:

- **Share** — copy and redistribute the material in any medium or format;
- **Adapt** — remix, transform, and build upon the material;

**Under the following terms:**

- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes.
- **Scientific Use Only** — Use of this dataset is restricted to non-commercial **scientific research and educational purposes**.

For more information, please refer to the full license text: [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)


## Our team:
- Walid Al-haidri ([Project Manager, ML Engineer](https://github.com/orgs/MRI-algorithms-and-methods/people/Walhaidri))
- Ekaterina Brui ([Project Co-manager, Medical physicist]())
- Zilia Badrieva ([Medical physicist](https://github.com/ZilyaB))
- Anatolii Levchuk ([ML Engineer,  Biomedical engineer](https://github.com/LeTond))
- Ksenia Belousova ([ML Engineer](https://github.com/Kseniyabel))
- Iuliia Pisareva ([Software Engineer](https://github.com/zi2p))
- Nikita Babich ([Software Engineer](https://github.com/spacexerq))
- Anna Konanykhina ([ML Engineer]())

