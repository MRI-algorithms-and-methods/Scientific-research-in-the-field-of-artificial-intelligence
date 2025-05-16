# 🧠 Pipeline Overview: Synthetic MRI Dataset Generation (Non AI)

![image](https://github.com/user-attachments/assets/c2888572-c93d-4d52-bd8e-c6e5373bd177)



## 1. Digital Phantom Creation

Utilize anatomically realistic digital phantoms, such as **BigBrain-MR**, which offer high-resolution representations of human anatomy with detailed MRI properties. [Source](https://www.sciencedirect.com/science/article/pii/S1053811923002203)

![BigBrain-MR Phantom](https://www.sciencedirect.com/science/article/pii/S1053811923002203)

## 2. Pulse Sequence Design

Design MRI pulse sequences using the **Pulseq** framework, allowing for hardware-independent and vendor-neutral sequence prototyping. [Source](https://pulseq.github.io/)


## 3. MRI Simulation with KomaMRI.jl

Employ **KomaMRI.jl**, an open-source MRI simulation framework developed in Julia, compatible with Pulseq sequences. KomaMRI.jl efficiently solves the Bloch equations using CPU and GPU parallelization, enabling realistic simulation of MRI acquisitions. [Source](https://pubmed.ncbi.nlm.nih.gov/36877139/)

![KomaMRI Simulation](https://onlinelibrary.wiley.com/doi/10.1002/mrm.29635)

## 4. Dockerized Virtual MRI Scanner

Implement the simulation environment within a Docker container, ensuring reproducibility and ease of deployment across different systems. This encapsulation allows for consistent simulation setups and facilitates integration into various workflows.

## 5. K-Space Data Generation and Sorting

Simulate raw MRI data acquisition, resulting in k-space data stored in the ISMRMRD format. Subsequently, sort the k-space data to prepare it for image reconstruction processes.

## 6. Image Reconstruction

Reconstruct MR images from the sorted k-space data using appropriate reconstruction algorithms, such as those provided by **MRIReco.jl**, ensuring high-fidelity image outputs. [Source](https://pubmed.ncbi.nlm.nih.gov/33817833/)

![MRI Reconstruction](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.29384)


