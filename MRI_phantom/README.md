# MRI_phantom
<p align="center">
  <img src="https://github.com/user-attachments/assets/2afdd2ac-2a6e-4596-b5a0-879b6220d66b" alt="image" />
</p>

A digital MRI phantom represents a set of various tissue characteristics, such as T1 and T2 relaxation times and proton density. Digital phantoms simulate the properties and behavior of different tissues during an MRI procedure. They are an essential tool for developers and researchers in the field of MRI, allowing them to test and optimize imaging techniques, pulse sequences, and reconstruction algorithms without the need for physical phantoms, volunteer scanning, or actual MRI procedures.


This repository contains code for generating quantitative maps of various modalities (T1, T2, T2*, proton density) and assembling them into phantoms in HDF5 format, and code for reconstructing raw MRI data and obtaining a weighted image.
Here’s a concise version of your text:



MRI brain digital phantoms were generated using anatomical brain masks from the [BrainWeb database](https://brainweb.bic.mni.mcgill.ca/brainweb/). Each slice was segmented into 11 tissue types, including:

- Gray matter  
- White matter  
- Cerebrospinal fluid (CSF)  
- Fat  
- Muscle  
- Bone marrow  
- Skin  
- Dura mater  
- Vessels

Each tissue mask was assigned characteristic MRI properties—proton density (PD), T1, and T2 relaxation times—based on values reported in the literature. This resulted in realistic, ground truth digital phantoms suitable for simulating MRI acquisitions.



