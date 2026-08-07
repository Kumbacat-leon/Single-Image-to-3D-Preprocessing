\# Single-Image-to-3D Preprocessing



Automated image pre-processing and evaluation pipeline for improving single-image-to-3D generation with TripoSR.



\## Overview



This project investigates whether automated image pre-processing can improve the quality and consistency of single-image-to-3D reconstruction without modifying the underlying 3D generation model.



The pipeline applies image processing operations before passing the input image to TripoSR, followed by quantitative evaluation of the generated 3D meshes.




## Pipeline



```text

Input Image

   ↓

Background Removal

   ↓

Object-Centered Cropping

   ↓

Adaptive Padding / Object Scaling

   ↓

Image Enhancement

   ↓

TripoSR

   ↓

3D Mesh

   ↓

Evaluation
