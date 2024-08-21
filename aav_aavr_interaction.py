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

def assemble_timeseries_array(u: mda.core.universe.Universe, selection: str) -> np.ndarray:
    '''
    This will create an array of the trajecotry's length.
    Each bead will be described by [x,y,z,resid]
    '''
    series = []
    for t in u.trajectory:
        p = u.select_atoms(selection).positions
        id = u.select_atoms(selection).resids.reshape(len(p),1)
        positions_and_ids = np.concatenate((p,id), axis=1)
        series.append(positions_and_ids)
    series = np.array(series)
    return series

def assemble_timeseries_array_with_masses(u: mda.core.universe.Universe, selection: str) -> np.ndarray:
    '''
    This will create an array of the trajecotry's length.
    Each bead will be described by array[x,y,z,resid,mass]
    '''
    series = []
    for t in u.trajectory:
        p = u.select_atoms(selection).positions
        id = u.select_atoms(selection).resids.reshape(len(p),1)
        mass = u.select_atoms(selection).masses.reshape(len(p),1)
        positions_ids_masses = np.concatenate((p,id,mass), axis=1)
        series.append(positions_ids_masses)
    series = np.array(series)
    return series


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


def compute_cog_ts(timeseries: np.ndarray) -> np.ndarray:
    init_frame = timeseries[0]
    resids = np.array(list(set(init_frame[:,3])))
    coords_ts = np.zeros((len(timeseries), len(resids), 3))
    for i in range(len(timeseries)):
        frame_coords = compute_cog(timeseries[i])
        coords_ts[i] += frame_coords
    return coords_ts


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


def compute_com_ts(timeseries: np.ndarray) -> np.ndarray:
    init_frame = timeseries[0]
    resids = np.array(list(set(init_frame[:,3])))
    coords_ts = np.zeros((len(timeseries), len(resids), 3))
    for i in range(len(timeseries)):
        frame_coords = compute_com(timeseries[i])
        coords_ts[i] += frame_coords
    return coords_ts
## Useful function to process the whole timeseries
def multiprocess_ts(func: callable, timeseries: np.ndarray) -> np.ndarray:
    with Pool(processes=cpu_count()) as pool:
        res = pool.map(func, [timeseries[0:1001],
                            timeseries[1001:2001],
                            timeseries[2001:3001],
                            timeseries[3001:4001],
                            timeseries[4001:5001],
                            timeseries[5001:6001],
                            timeseries[6001:7001],
                            timeseries[7001:8001],
                            timeseries[8001:9001],
                            timeseries[9001:10001]])
    results = np.concatenate((res[0],res[1],res[2],res[3],res[4],res[5],res[6],res[7],res[8],res[9]))
    return results

def scipy_cdist_ts(timeseries_A: np.ndarray, timeseries_B: np.ndarray) -> np.ndarray:
    init_array = np.zeros((len(timeseries_A), len(timeseries_A[0]), len(timeseries_B[0])), dtype=np.float16)
    for i in range(len(timeseries_A)):
        dist_array = cdist(timeseries_A[i], timeseries_B[i])
        init_array[i] += dist_array
    return init_array

def multiprocess_scipy_cdist_ts(func: callable, timeseries_A: np.ndarray, timeseries_B: np.ndarray) -> np.ndarray:
    with Pool(processes=cpu_count()) as pool:
        res = pool.starmap(func, [(timeseries_A[0:1000],timeseries_B[0:1000]),
                                  (timeseries_A[1000:2000],timeseries_B[1000:2000]),
                                  (timeseries_A[2000:3000],timeseries_B[2000:3000]),
                                  (timeseries_A[3000:4000],timeseries_B[3000:4000]),
                                  (timeseries_A[4000:5000],timeseries_B[4000:5000]),
                                  (timeseries_A[5000:6000],timeseries_B[5000:6000]),
                                  (timeseries_A[6000:7000],timeseries_B[6000:7000]),
                                  (timeseries_A[7000:8000],timeseries_B[7000:8000]),
                                  (timeseries_A[8000:9000],timeseries_B[8000:9000]),
                                  (timeseries_A[9000:10001],timeseries_B[9000:10001])])
        results = np.concatenate((res[0],res[1],res[2],res[3],res[4],res[5],res[6],res[7],res[8],res[9]))
        return results

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

def numba_cdist_ts(timeseries_A: np.ndarray, timeseries_B: np.ndarray) -> np.ndarray:
    init_array = np.zeros((len(timeseries_A), len(timeseries_A[0]), len(timeseries_B[0])), dtype=np.float16)
    for i in range(len(timeseries_A)):
        dist_array = numba_cdist(timeseries_A[i], timeseries_B[i])
        init_array[i] += dist_array
    return init_array