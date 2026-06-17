# CellSurf
A framework implementation to of 3D methods for data representation, surface reconstruction and geodesic distance measures built on top of the Open3D and rustworkx libraries. 

# Installation - For Users
To install CellSurf, you can use pip:
```bash
pip install cellsurf
```
# Installation - For Contributors
If you want to contribute to the development of CellSurf, you can clone the repository and install the package in editable mode. This allows you to make changes to the code and see the effects immediately without needing to reinstall the package.

## Pip
```bash
# Clone the repository
git clone https://github.com/Zatyrus/CellSurf.git
cd CellSurf

# Ideally create a new (conda) environment
# for this project to avoid conflicts with other packages
conda create -n minflux_spt python==3.12
conda activate MinSPT

# Install the package in editable mode
pip install -e .
```

## Python Poetry
```bash
# Clone the repository
git clone https://github.com/Zatyrus/CellSurf.git
cd CellSurf

# Ideally create a new (conda) environment
# for this project to avoid conflicts with other packages
conda create -n minflux_spt python==3.12
conda activate MinSPT

# Install the package using Poetry
poetry install
```
## Tips for Jupyter Notebooks
If you want to run the Jupyter notebooks included in the repository, you need to ensure that you have Jupyter installed in the same environment where CellSurf is installed. Here are the steps to set up the environment for running the notebooks:
```bash
# optionally activate the conda environment if you created one
conda activate MinSPT

# Install Jupyter and related packages
pip install jupyter ipykernel ipywidgets
```

# Outline of the Codebase
```bash
CellSurf - Repository
├───cellsurf
│   ├───core
│   │   ├───containers
│   │   ├───framework
│   │   ├───helpers
│   │   ├───util
│   ├───generators
│   └────visualization
├───test_data
└───tutorials
```

The CellSurf codebase is organized into several modules, each responsible for different aspects of the framework:
- `core`: Contains the core classes and functions for data representation, surface reconstruction, and geodesic distance measures. This includes containers for point clouds and meshes, the main framework for processing data, helper functions for various tasks, and utility functions for common operations.
- `generators`: Contains functions for generating synthetic 3D shapes and point clouds, which can be used for testing and demonstration purposes.
- `visualization`: Contains functions for visualizing point clouds and meshes using Open3D.

# Quick Start and Tutorials
To get started with CellSurf, import a generator for your favorite 3D-shape from the `cellsurf.generators` module and create a surface from it. For example, to create a surface from a sphere generator:
```python
from cellsurf.generators import generate_sphere, add_gaussian_noise
from cellsurf import GeometryTransformer

# Generate a sphere point cloud
sphere = generate_sphere(
   center = (0,0,0),
   radius=1.0, 
   num_points=1000
   )

# Add Gaussian noise to the point cloud
sphere = add_gaussian_noise(sphere, sigma=0.1)

# Let's take a look
sphere.visualize()

# Create a surface from the point cloud
sphere_mesh = GeometryTransformer().concave_hull_from(sphere, alpha = 0.8)

# color the point cloud and surface for better visualization
sphere.paint_uniform_color([0, 1, 0]) # green
sphere_mesh.paint_uniform_color([1, 0, 0]) # red

# Visualize the surface together with the original point cloud
# Tip: Press the "W" key in the visualizer to show the edges of the surface mesh
sphere_mesh.visualize_with(sphere)
```
For more detailed tutorials and examples, please refer to the Jupyter notebooks included in the `tutorials` directory of the repository. These notebooks cover various topics such as surface reconstruction, geodesic distance measures, and more.