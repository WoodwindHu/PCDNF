import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys
import random
import torch
from tqdm import tqdm 

class NormalizeUnitSphere(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def normalize(pcl, center=None, scale=None):
        """
        Args:
            pcl:  The point cloud to be normalized, (N, 3)
        """
        if center is None:
            p_max = pcl.max(dim=0, keepdim=True)[0]
            p_min = pcl.min(dim=0, keepdim=True)[0]
            center = (p_max + p_min) / 2    # (1, 3)
        pcl = pcl - center
        if scale is None:
            scale = (pcl ** 2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]  # (1, 1)
        pcl = pcl / scale
        return pcl, center, scale

    def __call__(self, data):
        assert 'pcl_noisy' not in data, 'Point clouds must be normalized before applying noise perturbation.'
        data['pcl_clean'], center, scale = self.normalize(data['pcl_clean'])
        data['center'] = center
        data['scale'] = scale
        return data

class AddNoise(object):
    def __init__(self, noise_std):
        super().__init__()
        self.noise_std = noise_std

    def __call__(self, data):
        data['pcl_noisy'] = data['pcl_clean'] + torch.randn_like(data['pcl_clean']) * self.noise_std
        data['noise_std'] = self.noise_std
        return data

def read_xyz(file_path):
    """
    Read points from an .xyz file.

    Parameters:
    file_path (str): Path to the .xyz file.

    Returns:
    numpy.ndarray: Nx3 array of points.
    """
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y, z = map(float, line.split())
            points.append([x, y, z])
    return np.array(points)

def compute_normals(points, k=30):
    """
    Compute normals for a point cloud using PCA.

    Parameters:
    points (numpy.ndarray): Nx3 array of points.
    k (int): Number of nearest neighbors to use for each point.

    Returns:
    numpy.ndarray: Nx3 array of normal vectors.
    """
    normals = np.zeros(points.shape)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)
    
    for i in range(points.shape[0]):
        neighbors = points[indices[i]]
        covariance_matrix = np.cov(neighbors, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
        normal = eigvecs[:, 0]  # The normal is the eigenvector corresponding to the smallest eigenvalue
        # let normal point against the original point
        if np.dot(normal, points[i]) < 0:
            normal = -normal
        normals[i] = normal
    
    return normals

def save_to_npy(points, normals, output_file):
    """
    Save points and normals to an .npy file.

    Parameters:
    points (numpy.ndarray): Nx3 array of points.
    normals (numpy.ndarray): Nx3 array of normal vectors.
    output_file (str): Path to the output .npy file.
    """
    data = np.hstack((points, normals))
    np.save(output_file, data)

input_path = sys.argv[1]
output_path = sys.argv[2]
os.makedirs(output_path, exist_ok=True)
input_files = os.listdir(input_path)
for file in tqdm(input_files):
    if file.endswith('.xyz'):
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file.replace('.xyz', f'.npy'))
        points = read_xyz(input_file)
        normals = compute_normals(points, k=30)
        save_to_npy(points, normals, output_file)
        # print(f"Saved points with normals to {output_file}")
# for noise in [0.0025, 0.005, 0.01, 0.015, 0.025]:
#     for file in tqdm(input_files):
#         if file.endswith('.xyz'):
#             input_file = os.path.join(input_path, file)
#             output_file = os.path.join(output_path, file.replace('.xyz', f'_normal_{noise}.npy'))
#             points = read_xyz(input_file)
#             data = {'pcl_clean': torch.tensor(points).float()}
#             data = NormalizeUnitSphere()(data)
#             data = AddNoise(noise)(data)
#             points = data['pcl_noisy'].numpy() * data['scale'].numpy() + data['center'].numpy()
#             normals = compute_normals(points, k=30)
#             save_to_npy(points, normals, output_file)
            # print(f"Saved points with normals to {output_file}")
