![Made with ❤️](https://img.shields.io/badge/AI%20-%E2%9D%A4-golden)
[![Made with ❤️](https://img.shields.io/badge/ITMO%20-%E2%9D%A4-red)](https://itmo.ru/)
[![Made with ❤️](https://img.shields.io/badge/Our%20group%20-%E2%9D%A4-blue)](https://physics.itmo.ru/en/research-group/5202)
[![Dataset](https://img.shields.io/badge/Data-Available-brightgreen.svg)](https://drive.google.com/file/d/1cIJ-pzP3Sd16GqXRqOqlJ0tyiZ0slTiW/view?usp=sharing)



# Generation of realistic physics-based synthetic magnetic resonance imaging data


![image](https://github.com/user-attachments/assets/84871bd0-548e-4f7b-8ee6-e6345b08c95b)

# Overview

The project focuses on developing a platform for generating realistic synthetic MRI data, which can be used for various applied tasks. 

The platform consists of several blocks: 

- **Data Collection Block** — Collects real MRI data with annotations, tissue physical parameters, and information about magnetic field inhomogeneities, noise, and artifacts.
  
- **[Digital Phantom Creation Block ](https://github.com/MRI-algorithms-and-methods/Scientific-research-in-the-field-of-artificial-intelligence/tree/main/MRI_phantom)** — Using the collected data, realiatic digital phantoms are created. We developed  and AI models (diffusion-based, GAN-based) for generation realistic digital MR phantoms.
  
- **[MRI Data Synthesis Block(pipeline)](https://github.com/MRI-algorithms-and-methods/Scientific-research-in-the-field-of-artificial-intelligence/tree/main/pipeline)** — Synthesizing additional MRI data using an MRI scanner simulator and pulse sequences developed by the team. For  an example of such synthetic data refere to **[Generated_dataset](https://github.com/MRI-algorithms-and-methods/Scientific-research-in-the-field-of-artificial-intelligence/tree/main/Generated_dataset)** used for training **[DL-based  Bloch simulator](https://github.com/MRI-algorithms-and-methods/Scientific-research-in-the-field-of-artificial-intelligence/tree/main/DL-based_Bloch_simulator)**
  
- **Validation Block** — Testing the platform on applied tasks, such as:
    - **[Enhancing of segmentation quality of  MR image by synthetic data](https://github.com/MRI-algorithms-and-methods/Scientific-research-in-the-field-of-artificial-intelligence/tree/main/brain_segmentation_lib)**.
    -  MR image reconstruction for ultra-low-field MRI using deep learning algorithms (upcoming)

The project involves collaboration with NIITFA and JET LAB LLC, as well as close interaction with the international scientific community.


## AI-powered approach for generation realistic digital MRI brain phantom:

## Our team:
- Walid Al-haidri ([Project Manager, ML Engineer]())
- Ekaterina Brui ([]())
- Zilia Badrieva ([]())
- Anatolii Levchuk ([](https://github.com/LeTond))
- Ksenia Belousova ([ML Engineer](https://github.com/Kseniyabel))
- Iuliia Pisareva ([Software Engineer](https://github.com/zi2p))
- Nikita Babich ([](https://github.com/spacexerq))
- Anna Konanykhina ([ML Engineer]())

