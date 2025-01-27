# CapMemUtils
 A collection of tools to analyse coarse-grained molecular dynamics (cgMD) trajectories involving (1) capsid, (2) receptor, and (3) membrane. All of this code in this reporitory has been used for analysing cgMD simulations of AAV2 capsid and AAVR in a model membrane.
## Datasets
Datasets are available here: https://doi.org/10.5281/zenodo.13361980 (AAV2-AAVR and AAV2-only membrane systems), https://doi.org/10.5281/zenodo.13362257 (AAVR-only membrane and solution state systems)

This includes 10 $\mu s$ simulations of (1) AAV2-AAVR system, (2) AAV2-only system, (3) AAVR-only system, and (4) the solution state system. All waters and ions have been removed from the final xtc files for convenience in data handling.

Results from the analyses conducted as described in the paper are also available here: https://doi.org/10.5281/zenodo.13359300

This includes all figure and compressed raw analysis outputs (in either .npy or .npz formats).
## Scripts
Main modules are aav_aavr_interaction.py and aav_lipid_cluster.py. These two modules include main utilities such as organising the trajectories, subsetting for only the beads of interest, calculating the contacts, and visualisation tools. Usage of these tools can be seen in jupyter notebooks in `notebooks` directory. Code for orientation analysis however is not in these teo modules and has been written separately in `notebooks/orientation.ipynb`.
