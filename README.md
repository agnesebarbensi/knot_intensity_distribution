# knot_intensity_distribution

This repository contains the accompanying information and software for the paper 
**Knot intensity distribution: a local measure of entanglement** by A.Barbensi and D.Celoria

## Organisation
The code and data in the repository are organised as follows:
There are 4 Jupyter notebooks, showcasing results from specific subsections of the paper (each is documented internally):
 
- **Compute_knot_intensity.ipynb** main notebook, explains how the knot intensity distributions for PL curves are performed

- **Ideal_Knots.ipynb** methodology and results for the analysis of the intensity distribution on ideal embeddings of knots with low crossing number

- **Random_knots.ipynb** methodology and results for the analysis of the intensity distribution on randomly generated knots

- **Crank_shaft_moves.ipynb** methodology and results for the cosmetic vs non-cosmetic strand passages 

In addition to datasets in various formats (pickled data, csv, npy), there are 4 Python files containing the functions used throughout the analysis in the notebooks (Functions_utils.py, Ideal_utils.py, Plotting_utils.py, Random_utils.py); the relevant ones are automatically imported by the notebooks.  

## Requirements
* Python 3
* Knoto-ID (Dorier, Julien, et al. "Knoto-ID: a tool to study the entanglement of open protein chains using the concept of knotoids." Bioinformatics (2018)): see https://github.com/sib-swiss/Knoto-ID

