import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from timeseries import *
from scipy.optimize import curve_fit
import os
import re

class Protein(Timeseries):
    def __init__(self, selection, path_gro, path_xtc):
        super().__init__(selection, path_gro, path_xtc)


class Membrane(Timeseries):

    martini_lipids = [
        "DAPC",
        "DBPC",
        "DFPC",
        "DGPC",
        "DIPC",
        "DLPC",
        "DNPC",
        "DOPC",
        "DPPC",
        "DRPC",
        "DTPC",
        "DVPC",
        "DXPC",
        "DYPC",
        "LPPC",
        "PAPC",
        "PEPC",
        "PGPC",
        "PIPC",
        "POPC",
        "PRPC",
        "PUPC",
        "DAPE",
        "DBPE",
        "DFPE",
        "DGPE",
        "DIPE",
        "DLPE",
        "DNPE",
        "DOPE",
        "DPPE",
        "DRPE",
        "DTPE",
        "DUPE",
        "DVPE",
        "DXPE",
        "DYPE",
        "LPPE",
        "PAPE",
        "PGPE",
        "PIPE",
        "POPE",
        "PQPE",
        "PRPE",
        "PUPE",
        "DAPS",
        "DBPS",
        "DFPS",
        "DGPS",
        "DIPS",
        "DLPS",
        "DNPS",
        "DOPS",
        "DPPS",
        "DRPS",
        "DTPS",
        "DUPS",
        "DVPS",
        "DXPS",
        "DYPS",
        "LPPS",
        "PAPS",
        "PGPS",
        "PIPS",
        "POPS",
        "PQPS",
        "PRPS",
        "PUPS",
        "DAPA",
        "DBPA",
        "DFPA",
        "DGPA",
        "DIPA",
        "DLPA",
        "DNPA",
        "DOPA",
        "DPPA",
        "DRPA",
        "DTPA",
        "DVPA",
        "DXPA",
        "DYPA",
        "LPPA",
        "PAPA",
        "PGPA",
        "PIPA",
        "POPA",
        "PRPA",
        "PUPA",
        "DAPG",
        "DBPG",
        "DFPG",
        "DGPG",
        "DIPG",
        "DLPG",
        "DNPG",
        "DOPG",
        "DPPG",
        "DRPG",
        "DTPG",
        "DVPG",
        "DXPG",
        "DYPG",
        "LPPG",
        "PAPG",
        "PGPG",
        "PIPG",
        "POPG",
        "PRPG",
        "PIP2"
    ]
    upper_start = np.NaN
    upper_end = np.NaN
    lower_start = np.NaN
    lower_end = np.NaN

    def __init__(self, selection, path_gro, path_xtc, repeat, layer, num_lipid_residues_upper, num_lipid_residues_lower):
        #### Need to restrict selection to only MATINI lipid beads ####
        if not re.match(r'resname ' + '[A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9]', selection):
            raise ValueError("Error! pattern is not recognised. Make sure you use e.g., selection='resname POPC'")
        if selection.strip("resname ") not in Membrane.martini_lipids:
            raise ValueError("Invalid names entered. Maybe the lipid is not in the Martini Database? Please try again with e.g. 'resname POPC'")
        super().__init__(selection, path_gro, path_xtc)
        self.path_gro = path_gro
        self.path_xtc = path_xtc
        if isinstance(repeat, int) == False:
            raise ValueError("Number of repeat can only be an integer.")
        self.repeat = repeat ## This is solely for bookkeeping purposes.
        if layer not in ["upper", "lower"]:
            raise ValueError("Sorry, value for lipid bilayer can only be 'upper' or 'lower'.")
        self.layer = layer
        if isinstance(num_lipid_residues_upper, int) == False:
            raise ValueError("Number of lipid residues in the upper leaflet can only be an integer.")
        if isinstance(num_lipid_residues_lower, int) == False:
            raise ValueError("Number of lipid residues in the lower leaflet can only be an integer.")
        self.num_lipid_residues_upper = num_lipid_residues_upper
        self.num_lipid_residues_lower = num_lipid_residues_lower
        ## Note that the membrane timeseries is not assembled yet.
    
    def get_membrane_layer_indices(self):
        ### Assuming no water in the processed gro and xtc files.
        ### Find the easiest pieces of information first..
        Membrane.upper_start = self.universe.select_atoms("not protein").resids[0]
        Membrane.lower_end = self.universe.select_atoms("not protein").resids[-1]
        Membrane.upper_end = Membrane.upper_start + self.num_lipid_residues_upper - 1 # Subtract 1 Because MDAnalysis selection is inclusive on both ends
        Membrane.lower_start = Membrane.upper_end +  1
        return None
    
    def assemble_timeseries_membrane(self):
        if self.layer == "upper":
            self.selection = self.selection + f" and resid {Membrane.upper_start}:{Membrane.upper_end}"
            self.assemble_timeseries_array()
        if self.layer == "lower":
            self.selection = self.selection + f" and resid {Membrane.lower_start}:{Membrane.lower_end}"
            self.assemble_timeseries_array()
        return None
    
    def get_layer_reference_phosphates(self):
        if self.layer == "upper":
            PO4_ref = Timeseries(selection = f"name PO4 and resid {Membrane.upper_start}:{Membrane.upper_end}", path_gro = self.path_gro, 
                                    path_xtc = self.path_xtc)
            PO4_ref.assemble_timeseries_array()
            self.layer_reference_phosphates = PO4_ref.data
        
        elif self.layer == "lower":
            PO4_ref = Timeseries(selection = f"name PO4 and resid {Membrane.lower_start}:{Membrane.lower_end}", path_gro = self.path_gro, 
                                    path_xtc = self.path_xtc)
            PO4_ref.assemble_timeseries_array()
            self.layer_reference_phosphates = PO4_ref.data
        return None
    
    def acquire_membrane_data(self):
        self.get_membrane_layer_indices()
        self.get_layer_reference_phosphates()
        self.assemble_timeseries_membrane()
        return None
        
    ############ Cluster Analysis ############
    ############ Cluster Analysis ############
    ############ Cluster Analysis ############
    ############ Cluster Analysis ############
    ############ Cluster Analysis ############
    
    def compute_clusters(self, frame, eps, ms):
        data_ = self.data[frame]
        list_of_lipid_resids = np.array(list(set(data_[:,3])))
        lipid_coords = []
        for i in range(len(list_of_lipid_resids)):
                resid = list_of_lipid_resids[i]
                positions = data_[np.where(data_[:,3] == resid)][:,0:3]
                positions_cog = positions.mean(axis=0)
                lipid_coords.append(positions_cog)
        lipid_coords = np.array(lipid_coords)
        model = DBSCAN(eps=eps, min_samples=ms)
        model.fit_predict(lipid_coords)
        return model

    @staticmethod
    def compute_average_clustsize(model, exclude_noise = True):
        cluster_ids = np.array(list(set(model.labels_)))
        ## Remove Noise Group
        if exclude_noise:
            if -1 in cluster_ids:
                cluster_ids.sort()
                cluster_ids = np.delete(cluster_ids, 0, 0)
            else:
                pass
        cluster_ids_count  = []
        for i in range(len(cluster_ids)):
            count = np.where(model.labels_ == cluster_ids[i])
            count = np.array(count)
            cluster_ids_count.append(count.shape[1])
        avclustsize = np.array(cluster_ids_count)
        avclustsize = avclustsize.mean()
        return avclustsize

    def clust_ts(self, eps, ms, exclude_noise = True):
        Lipids_of_Interest_nclust = []
        Lipids_of_Interest_clustsize = []
        
        for i in range(len(self.data)):
            model = self.compute_clusters(frame = i, eps = eps, ms = ms)
            if -1 not in set(model.labels_):
                nclust = len(set(model.labels_))
            elif exclude_noise:
                if -1 in set(model.labels_):
                    nclust = len(set(model.labels_)) - 1 ## exclude noise group
            Lipids_of_Interest_nclust.append(nclust)
            clustsize = Membrane.compute_average_clustsize(model)
            Lipids_of_Interest_clustsize.append(clustsize)
        Lipids_of_Interest_nclust = np.array(Lipids_of_Interest_nclust)
        Lipids_of_Interest_clustsize = np.array(Lipids_of_Interest_clustsize)

        return Lipids_of_Interest_nclust, Lipids_of_Interest_clustsize
    
    def run_cluster_dbscan(self, eps, ms, exclude_noise = True):
        Lipids_of_Interest_nclust, Lipids_of_Interest_clustsize = self.clust_ts(eps, ms, exclude_noise = exclude_noise)
        if not os.path.isdir("lipid_clusters"):
            os.makedirs("lipid_clusters")
        np.savez(f"lipid_clusters/nclust_repeat{self.repeat}_{self.name}_layer{self.layer}.npz", Lipids_of_Interest_nclust)
        np.savez(f"lipid_clusters/clustsize_repeat{self.repeat}_{self.name}_layer{self.layer}.npz", Lipids_of_Interest_clustsize)
        return None
    
    ############ Xcorr Analysis ############
    ############ Xcorr Analysis ############
    ############ Xcorr Analysis ############
    ############ Xcorr Analysis ############
    ############ Xcorr Analysis ############

    @staticmethod
    def map_to_grid(coord, grid_width, grid_height, grid_size):
        x_index = int(coord[0] // grid_width)
        y_index = int(coord[1] // grid_height)
        return x_index + y_index * grid_size

    def xcorr(self, side_length = 18):
        ## initiate target results lists    
        lipids_count_matrix = []
        zdev_list_matrix = []
        xcorr_matrix = []
        self.get_layer_reference_phosphates()

        x_min = min(self.layer_reference_phosphates[:,:,0].min(), self.data[:,:,0].min())
        x_max = max(self.layer_reference_phosphates[:,:,0].max(), self.data[:,:,0].max())
        y_min = min(self.layer_reference_phosphates[:,:,1].min(), self.data[:,:,1].min())
        y_max = max(self.layer_reference_phosphates[:,:,1].max(), self.data[:,:,1].max())

        for frame in range(len(self.data)):
            lipid = self.data[frame]
            layer = self.layer_reference_phosphates[frame]
            # Define grid dimensions
            side_length = side_length
            grid_width = grid_height = side_length 

            ## Grid Size = The number of grid squares along one axis.
            grid_size = max(x_max - x_min, y_max - y_min) // side_length ## Choose maximum to cover all possible squares.

            # Generate bead xy coordinates
            bead_coordinates = lipid[:,0:2] + abs(min(x_min,y_min)) ## Get rid of the negatives
            # Map bead coordinates to grid indices
            bead_indices = np.apply_along_axis(Membrane.map_to_grid, 1, bead_coordinates, grid_width, grid_height, grid_size)
            lipid_indexed = np.concatenate((lipid,bead_indices.reshape((len(bead_coordinates),1))), axis=1, dtype=float)

            ## Repeat with the PO4 beads 

            # Generate bead xy coordinates
            bead_coordinates = layer[:,0:2] + abs(min(x_min,y_min)) ## Get rid of the negatives
            # Map bead coordinates to grid indices
            bead_indices = np.apply_along_axis(Membrane.map_to_grid, 1, bead_coordinates, grid_width, grid_height, grid_size)
            layer_indexed = np.concatenate((layer,bead_indices.reshape((len(bead_coordinates),1))), axis=1, dtype=float)

            n_grid_squares = int(grid_size**2) ## Number of grid squares
            lipids_count = np.zeros((n_grid_squares,1))
            zdev_list = np.zeros((n_grid_squares,1))
            zmean = layer_indexed[:,2].mean()
            for i in range(n_grid_squares):
                data_to_analyse = lipid_indexed[np.where(layer_indexed[:,4] == i)]
                if len(data_to_analyse) == 0:
                    continue
                ## Count the number of the lipid molecules of interest
                lipids_count[i] += len(data_to_analyse)
                ## Calculate how much the average z position of the patch deviates from the average membrane position.
                z_positions = layer_indexed[np.where(layer_indexed[:,4] == i)][:,2]
                if len(z_positions) == 0:
                    continue
                ## If there is no PO4 bead in the grid square; assume no difference.
                zdev = z_positions.mean() - zmean
                zdev_list[i] += zdev
            ## Normalise

            lip_norm = (lipids_count - lipids_count.mean())/(np.std(lipids_count))
            z_norm = (zdev_list - zdev_list.mean())/(np.std(zdev_list))

            ## Cross-Correlation

            xcorr = np.correlate(lip_norm[:,0], z_norm[:,0]) / len(lip_norm)
            lipids_count_matrix.append(lipids_count)
            zdev_list_matrix.append(zdev_list)
            xcorr_matrix.append(xcorr)

        lipids_count_matrix =  np.array(lipids_count_matrix)
        zdev_list_matrix = np.array(zdev_list_matrix)
        xcorr_matrix = np.array(xcorr_matrix)
        return lipids_count_matrix, zdev_list_matrix, xcorr_matrix
    
    def run_lipid_position_curvature_xcorr(self):
        lipids_count_matrix, zdev_list_matrix, xcorr_matrix = self.xcorr()
        if not os.path.isdir("lipid_xcorr"):
            os.makedirs("lipid_xcorr")
        np.savez(f"lipid_xcorr/lipids_count_matrix_repeat{self.repeat}_{self.name}_layer{self.layer}.npz", lipids_count_matrix)
        np.savez(f"lipid_xcorr/zdev_list_matrix_repeat{self.repeat}_{self.name}_layer{self.layer}.npz", zdev_list_matrix)
        np.savez(f"lipid_xcorr/xcorr_matrix_repeat{self.repeat}_{self.name}_layer{self.layer}.npz", xcorr_matrix)
        return None
    
    ############ Curvature Analysis ############
    ############ Curvature Analysis ############
    ############ Curvature Analysis ############
    ############ Curvature Analysis ############
    ############ Curvature Analysis ############
    
    def func_to_opt(self, a, b, c, d, e, f, g, h):
        x = self.data[:,0]
        y = self.data[:,1]
        return np.array([a + b*(x) + c*(x**2) + d*(y) + e*x*y + f*(y**2) + g*(x**3) + h*(y**3)]).T[:,0]

    def get_curvature(self, params):
        x = self.data[:,0]
        y = self.data[:,1]
        a, b, c, d, e, f, g, h = params
        k_x = np.absolute(2*c + 6*g*x) / ((np.sqrt(1+(b +2*c*x + e*y + 3*g*x**2)**2))**3)
        k_y = np.absolute(2*f + 6*h*y) / ((np.sqrt(1+(d + e*x + 2*f*y + 3*h*y**2)**2))**3)
        K = k_x * k_y
        return K


####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
############################Construction Site#######################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
    ### Plot
    def plot_clust(Lipids_of_Interest_nclust,Lipids_of_Interest_clustsize, name):
        Lipids_of_Interest_nclust_rollingav = []
        for i in range(len(Lipids_of_Interest_nclust)):
            rollingav = Lipids_of_Interest_nclust[i:i+50].mean()
            Lipids_of_Interest_nclust_rollingav.append(rollingav)
        Lipids_of_Interest_nclust_rollingav = np.array(Lipids_of_Interest_nclust_rollingav)
        plt.plot(np.arange(0,10.001,0.001),Lipids_of_Interest_nclust, color = 'blue')
        plt.plot(np.arange(0,10.001,0.001),Lipids_of_Interest_nclust_rollingav, color = "cyan")
        plt.ylabel(f"Number of {name} Clusters")
        plt.xlabel("Time ($\mu$s)")
        plt.ylim(0,25)
        if not os.path.isdir("clust_results"):
            os.makedirs("clust_results")
        plt.savefig(f"clust_results/{name}_nclust.png",dpi=300)
        plt.show()

        Lipids_of_Interest_clustsize_rollingav = []
        for i in range(len(Lipids_of_Interest_clustsize)):
            rollingav = Lipids_of_Interest_clustsize[i:i+50].mean()
            Lipids_of_Interest_clustsize_rollingav.append(rollingav)
        Lipids_of_Interest_clustsize_rollingav = np.array(Lipids_of_Interest_clustsize_rollingav)

        plt.plot(np.arange(0,10.001,0.001),Lipids_of_Interest_clustsize, color = "lime")
        plt.plot(np.arange(0,10.001,0.001),Lipids_of_Interest_clustsize_rollingav, color = "darkgreen")
        plt.ylabel(f"Average Cluster Size\n(Number of {name} molecules within a cluster)")
        plt.xlabel("Time ($\mu$s)")
        plt.ylim(0,120)
        if not os.path.isdir("clust_results"):
            os.makedirs("clust_results")
        plt.savefig(f"clust_results/{name}_clustsize.png",dpi=300)
        plt.show()

    ## Cross-correlation between # of lipids and the shift in z of the membrane from the average membrane z (i.e. curvature)
    # Function to map bead coordinates to grid indices

    ### Plot
    def CCF_plot(xcorr_matrix, name):
        plt.plot(np.arange(0,10.001,0.001),xcorr_matrix,color='purple')
        plt.xlabel("Time ($\mu$s)")
        plt.ylabel(f"Cross-Correlation Coefficient\nbetween $\Delta$z and {name} lipid beads count")

        CC_rollingav = []
        for i in range(len(xcorr_matrix)):
            rollingav = xcorr_matrix[i:i+50].mean()
            CC_rollingav.append(rollingav)
        CC_rollingav = np.array(CC_rollingav)

        plt.plot(np.arange(0,10.001,0.001), CC_rollingav,color='magenta')
        plt.ylim(-0.5,0.5)
        if not os.path.isdir("ccf_results"):
            os.makedirs("ccf_results")
        plt.show()


## Analysis of Contact with the capsid

def capsid_lipid_contact(capsid, Lipids_of_Interest,cutoff=10):
    list_of_capsid_resids = np.array(list(set(capsid[0][:,3])))
    list_of_lipid_resids = np.array(list(set(Lipids_of_Interest[0][:,3])))
    capsid_lipid_contact_array_ts = []
    capsid_lipid_contact_array_count = np.zeros((len(list_of_capsid_resids),len(list_of_lipid_resids)))
    for ff in range(len(capsid)):

        ref = capsid[ff]
        lipid = Lipids_of_Interest[ff]
        ref_coords = []
        lipid_coords = []

        for i in range(len(list_of_capsid_resids)):
            resid = list_of_capsid_resids[i]
            positions = ref[np.where(ref[:,3] == resid)][:,0:3]
            positions_cog = positions.mean(axis=0)
            ref_coords.append(positions_cog)

        for i in range(len(list_of_lipid_resids)):
            resid = list_of_lipid_resids[i]
            positions = lipid[np.where(lipid[:,3] == resid)][:,0:3]
            positions_cog = positions.mean(axis=0)
            lipid_coords.append(positions_cog)

        ref_coords = np.array(ref_coords)
        lipid_coords = np.array(lipid_coords)

        capsid_lipid_dist_array = numba_cdist(ref_coords, lipid_coords)
        capsid_lipid_contact_array = capsid_lipid_dist_array < cutoff
        capsid_lipid_contact_array_ts.append(capsid_lipid_contact_array)
        capsid_lipid_contact_array_count += capsid_lipid_contact_array

    capsid_lipid_contact_array_ts = np.array(capsid_lipid_contact_array_ts)

    capsid_lipid_contact_freq = np.zeros(len(list_of_capsid_resids))

    for i in range(len(capsid_lipid_contact_array_count)):
        freq = capsid_lipid_contact_array_count[i].sum()
        capsid_lipid_contact_freq[i] += freq

    return capsid_lipid_contact_array_ts, capsid_lipid_contact_freq

### Write B factor list dat file for visualisation. DAT file format is compatible with gmx editconf.

def toBfact(capsid_lipid_contact_freq,b,name):
    maxb = 1
    capsid_lipid_contact_freq_frac = capsid_lipid_contact_freq*99/maxb ## B factor limit is 99 Angs squared
    res_range = np.arange(b,b+len(capsid_lipid_contact_freq_frac),1)
    lines = []
    lines.append(str(len(res_range)))
    for i in range(len(res_range)):
        ri = res_range[i]
        bf = capsid_lipid_contact_freq_frac[i]
        line = f"{ri} {bf}"
        lines.append(line)
    with open(f"{name}_bfact.dat", "w") as f:
        for line in lines:
            f.write(line + "\n")

def read_dat_as_occupancy(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        lines = lines[1:] ## Exclude residue number line
        occupancy_list =[]
        for line in lines:
            line = line.split()
            line = float(line[1])
            occupancy_list.append(line) ## Append Occupancy
        occupancy_list = np.array(occupancy_list)
        occupancy_list = occupancy_list / 99
        return occupancy_list
## Plot
## Only works with 2 pentamers
def plot_capsid_lipid(capsid_lipid_contact_freq, name):
    plt.figure(figsize=(10,7))
    plt.plot(np.arange(15511,len(capsid_lipid_contact_freq)+15511,1), capsid_lipid_contact_freq[:], color='b')
    plt.ylabel(f"Number of contacts with {name}")
    for i in range(30,40):
        plt.axvspan(43+i*517, 49+i*517, zorder=0, alpha=0.2, color='orange', label='VR-I')
        plt.axvspan(107+i*517, 111+i*517, zorder=0, alpha=0.2, color='red', label='VR-II')
        plt.axvspan(161+i*517, 169+i*517, zorder=0, alpha=0.2, color='pink', label='VR-III')
        plt.axvspan(230+i*517, 249+i*517, zorder=0, alpha=0.2, color='lime', label='VR-IV')
        plt.axvspan(268+i*517, 285+i*517, zorder=0, alpha=0.2, color='blue', label='VR-V')
        plt.axvspan(306+i*517, 322+i*517, zorder=0, alpha=0.2, color='magenta', label='VR-VI')
        plt.axvspan(325+i*517, 337+i*517, zorder=0, alpha=0.2, color='violet', label='VR-VII')
        plt.axvspan(360+i*517, 375+i*517, zorder=0, alpha=0.2, color='yellow', label='VR-VIII')
        plt.axvspan(485+i*517, 492+i*517, zorder=0, alpha=0.2, color='grey', label='VR-IX')
    plt.legend(['Frequency','VR-I','VR-II','VR-III','VR-IV','VR-V','VR-VI','VR-VII','VR-VIII','VR-IX'])
    plt.axvline(x=18096, color = 'k', linestyle='--')
    plt.text(x=16096,y=3500,s='capsomers 30-35',fontsize=16)
    plt.text(x=19096,y=3500,s='capsomers 35-40',fontsize=16)
    plt.xlabel("Capsid Residue")
    plt.ylim(0,7000)
    if not os.path.isdir("capsidlipidint_results"):
        os.makedirs("capsidlipidint_results")
    plt.savefig(f"capsidlipidint_results/{name}_capsid.png",dpi=300)

## For Lipid Time Series Distribution

def plot_timeseries(lipid, leaflet, name):

    colordict = {'dpg3': 'purple',
             'popcupper': 'darkgrey',
             'popeupper': 'grey',
             'dopcupper': 'green',
             'dopeupper': 'bisque',
             'cholupper': 'cyan',
             'pops': 'tan',
             'dops': 'lime',
             'pop2': 'pink',
             'popclower': 'darkgrey',
             'popelower': 'grey',
             'dopclower': 'green',
             'dopelower': 'bisque',
             'chollower': 'cyan'}
    
    fig, axs = plt.subplots(7, figsize=(8,45))
    timestamps = [0, 100, 1000, 2000, 4000, 8000, 10000]
    lipid_data = lipid
    leaflet_data = leaflet
    for i in range(len(timestamps)):
        ff = timestamps[i]
        data = lipid_data[ff]
        leaflet_ = leaflet_data[ff]
        x_data = data[:,0]
        y_data = data[:,1]
        x_leaflet_ = leaflet_[:,0]
        y_leaflet_ = leaflet_[:,1]
        z_leaflet_ = leaflet_[:,2]
        axs[i].scatter(x_data, y_data, color=colordict[str(name)], s=10, edgecolors= "black", linewidths=0.2, zorder=10)
        g = axs[i].tricontourf(x_leaflet_,y_leaflet_,z_leaflet_, vmin=70, vmax=360, zorder=5)
        axs[i].set_xlabel("X ($\AA$)")
        axs[i].set_ylabel("Y ($\AA$)")
        plt.colorbar(g, label= "z-height ($\AA$)")
        axs[i].set_title(f"{ff} ns")
    fig.tight_layout()
    if not os.path.isdir("lipid_2d"):
        os.makedirs("lipid_2d")
    plt.savefig(f"lipid_2d/{name}_ts.png",dpi=300)
    plt.show()
