# Microstructure Analysis with PuMA & Sailfish
This repository provides tools to compute microstructural properties and run fluid flow simulations on 3D porous media using PuMA
 and Sailfish.

It demonstrates how to extract key parameters such as specific surface area, mean intercept length, permeability, tortuosity, and also perform Lattice Boltzmann Method (LBM) simulations on image-based microstructures.

✨ Features
✅ Compute Specific Surface Area
✅ Compute Mean Intercept Length
✅ Compute Permeability (x, y, z)
✅ Compute Tortuosity (x, y, z)
✅ Estimate permeability using Kozeny–Carman equation
✅ Run 2D LBM simulations with Sailfish
✅ Generate synthetic sample data (binary 3D volume of "ice blobs")

📦 Requirements
Make sure you have the following installed:
Python 3.8+
NumPy
tifffile
PuMA

⚙️ Class Overview
Micro_Structure(image, ws)
Wrapper class for computing microstructural properties.
Specific_Surface_Area() → (Surface_Area, Specific_SurfaceArea)
Mean_Intercept_Length() → float
Permeability() → (kx, ky, kz)
Tortuosity() → (τx, τy, τz)
kozeny_carman_surface(porosity, C=5) → k
run_lbm(inlet_vel=0.05) → runs Sailfish simulation
Sailfish

(requires CUDA for GPU acceleration)
