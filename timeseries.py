import numpy as np
import MDAnalysis as mda
from multiprocessing import Pool, cpu_count
from scipy.spatial.distance import cdist
import numba as nb

'''

Data Structure Expected can be created by assemble_timeseries_array function.
Essentially, it is an array of selected bead positions for every frame.
In each frame, each entry is the vector describing [x, y, z, resid] for each bead. 
To condense the calculation, you can compute residue-wise centre of geometry (COG) with compute_cog,
or centre of mass (COM) with compute_com and calculate only pairwise residue-residue distances.
Note that some MARTINI beads' masses may not be recognised by MDAnalysis do their guessed masses are wrong (0).
Centre of mass-based calculation is more suitable for atomistic simulations.
Distance calculation functions are based on scipy cdist or numba (the latter is faster).

'''

class Timeseries:
    def __init__(self, selection: str, prefix_path: str):
        self.selection = selection
        self.prefix_path = prefix_path
        self.universe = mda.Universe(prefix_path + '.gro', prefix_path + '.xtc')
        self.data = np.array([])

    def assemble_timeseries_array(self) -> np.ndarray:
        '''
        This will create an array of the trajecotry's length.
        Each bead will be described by [x,y,z,resid]
        '''
        series = []
        u = self.universe
        for t in u.trajectory:
            p = u.select_atoms(self.selection).positions
            id = u.select_atoms(self.selection).resids.reshape(len(p),1)
            positions_and_ids = np.concatenate((p,id), axis=1)
            series.append(positions_and_ids)
        self.data = np.array(series)
        return None

    def assemble_timeseries_array_with_masses(self) -> np.ndarray:
        '''
        This will create an array of the trajecotry's length.
        Each bead will be described by array[x,y,z,resid,mass]
        '''
        series = []
        u = self.universe
        for t in u.trajectory:
            p = u.select_atoms(self.selection).positions
            id = u.select_atoms(self.selection).resids.reshape(len(p),1)
            mass = u.select_atoms(self.selection).masses.reshape(len(p),1)
            positions_ids_masses = np.concatenate((p,id,mass), axis=1)
            series.append(positions_ids_masses)
        self.data = np.array(series)
        return None

    @staticmethod
    def compute_cog(frame: np.ndarray) -> np.ndarray:
        '''
        Definition of COG: COG_residue_A = mean(position_a),
        where a = a bead that makes up residue A.
        '''
        resids = np.array(list(set(frame[:,3])))
        coords = np.zeros((len(resids),3), dtype=np.float16)
        for i in range(len(resids)):
            resid = resids[i]
            positions = frame[np.where(frame[:,3] == resid)][:,0:3]
            positions_cog = positions.mean(axis=0, dtype=np.float16)
            coords[i] += positions_cog
        return coords

    def compute_cog_ts(self) -> np.ndarray:
        init_frame = self.data[0]
        resids = np.array(list(set(init_frame[:,3])))
        coords_ts = np.zeros((len(self.data), len(resids), 3))
        for i in range(len(self.data)):
            frame_coords = Timeseries.compute_cog(self.data[i])
            coords_ts[i] += frame_coords
        return coords_ts

    @staticmethod
    def compute_com(frame: np.ndarray) -> np.ndarray:
        '''
        Definition of COM: COM_residue_A = sum(mass_a * position_a)/mass_A,
        where a = a bead that makes up residue A.
        '''
        resids = np.array(list(set(frame[:,3])))
        coords = np.zeros((len(resids),3), dtype=np.float16)
        for i in range(len(resids)):
            resid = resids[i]
            positions = frame[np.where(frame[:,3] == resid)][:,0:3]
            masses = frame[np.where(frame[:,3] == resid)][:,4].reshape(len(positions),1)
            positions_com = np.sum(positions * masses, axis=0)/np.sum(masses) ## Broadcast
            coords[i] += positions_com
        return coords


    def compute_com_ts(self) -> np.ndarray:
        init_frame = self.data[0]
        resids = np.array(list(set(init_frame[:,3])))
        coords_ts = np.zeros((len(self.data), len(resids), 3))
        for i in range(len(self.data)):
            frame_coords = Timeseries.compute_com(self.data[i])
            coords_ts[i] += frame_coords
        return coords_ts
    
    ## Useful function to process the whole timeseries
    def multiprocess_ts(self, func, num_chunks = 10):
        ## If blocks to catch nonsense
        if num_chunks < 0 or num_chunks == 0:
            raise ValueError("Number of processes cannot be less than or equal to 0.")
        if isinstance(num_chunks, int) == False:
            raise ValueError("Number of processes need to be an integer.")
        ## Prepare chunks
        chunk_length = int(np.floor(len(self.data) / num_chunks))
        chunk_ind = [i*chunk_length for i in range(num_chunks + 1)]
        chunks = [self.data[chunk_ind[j]: chunk_ind[j+1]] for j in range(len(chunk_ind))]
        if chunk_ind[-1] < len(self.data): ## add in extra bit
            chunks.append(self.data[chunk_ind[-1]: len(self.data)])
        ## Execute
        with Pool(processes=cpu_count()) as pool:
            res = pool.map(func, chunks)
        results = np.concatenate(res) ## does not work
        return results

    @staticmethod
    def scipy_cdist_ts(timeseries_A, timeseries_B) -> np.ndarray:
        init_array = np.zeros((len(timeseries_A.data), len(timeseries_A.data[0]), len(timeseries_B.data[0])), dtype=np.float16)
        for i in range(len(timeseries_A.data)):
            dist_array = cdist(timeseries_A.data[i], timeseries_B.data[i])
            init_array[i] += dist_array
        return init_array
    
    @staticmethod
    @nb.njit(fastmath=True,parallel=True)
    def numba_cdist(residues_1: np.ndarray,residues_2: np.ndarray) -> np.ndarray:
        '''
        This is the fastest pairwise distance calculation between two residues.
        Bare in mind that this function will output float32 (single precision),
        so if double precision is needed, please change dtype of res to nb.float64.
        But that will also cost double memory.
        Speed is already phenomenal without multiprocessing.
        '''
        res=np.empty((residues_1.shape[0],residues_2.shape[0]),dtype=nb.float32)
        for i in nb.prange(residues_1.shape[0]):
            for j in range(residues_2.shape[0]):
                res[i,j]=np.sqrt((residues_1[i,0]-residues_2[j,0])**2+(residues_1[i,1]-residues_2[j,1])**2+(residues_1[i,2]-residues_2[j,2])**2)
        return res

    @staticmethod
    def numba_cdist_ts(timeseries_A, timeseries_B) -> np.ndarray:
        init_array = np.zeros((len(timeseries_A.data), len(timeseries_A.data[0]), len(timeseries_B.data[0])), dtype=np.float16)
        for i in range(len(timeseries_A.data)):
            dist_array = Timeseries.numba_cdist(timeseries_A.data[i], timeseries_B.data[i])
            init_array[i] += dist_array
        return init_array
     
    @staticmethod
    def multiprocess_dist_ts(timeseries_A: np.ndarray, timeseries_B: np.ndarray, func: callable, num_chunks = 10) -> np.ndarray:
        ## If blocks to catch nonsense
        if num_chunks < 0 or num_chunks == 0:
            raise ValueError("Number of processes cannot be less than or equal to 0.")
        if isinstance(num_chunks, int) == False:
            raise ValueError("Number of processes need to be an integer.")
        if len(timeseries_A.data) != len(timeseries_B.data):
            raise ValueError("Number of frames should be the same for the two timeseries.")
        ## Prepare chunks
        chunk_length = int(np.floor(len(timeseries_A.data) / num_chunks))
        chunk_ind = [i*chunk_length for i in range(num_chunks + 1)]
        chunks = [(timeseries_A.data[chunk_ind[j]: chunk_ind[j+1]], timeseries_B.data[chunk_ind[j]: chunk_ind[j+1]]) for j in range(len(chunk_ind))]
        if chunk_ind[-1] < len(timeseries_A.data): ## add in extra bit
            chunks.append(
                (timeseries_A.data[chunk_ind[-1]: len(timeseries_A)], timeseries_B.data[chunk_ind[-1]: len(timeseries_B)])
                )
        ## Execute
        with Pool(processes=cpu_count()) as pool:
            res = pool.starmap(func, chunks)
            results = np.concatenate(res)
            return results