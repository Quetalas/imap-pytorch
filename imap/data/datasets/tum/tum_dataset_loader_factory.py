from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from imap.camera.camera import Camera
from .tum_dataset_loader import TUMDatasetLoader

DEFAULT_CAMERA_MATRIX = np.array([[525.0, 0, 319.5],
                                  [0, 525.0, 239.5],
                                  [0, 0, 1.]], dtype=np.float32)


class TUMDatasetLoaderFactory(object):
    @staticmethod
    def make_dataset_loader(dataset_path, scene_name, association_file_name, frame_indices,
                            camera_matrix=DEFAULT_CAMERA_MATRIX, distance_koef=1., clip_distance_threshold=4.,
                            factor=5000.):
        sequence_directory = Path(dataset_path) / scene_name
        association_file = sequence_directory / association_file_name
        print(f"Reading {association_file}")

        associations = pd.read_csv(association_file, names=[i for i in range(12)], sep=' ')
        positions = associations.iloc[:, 5:].values
        positions = np.array([TUMDatasetLoaderFactory.tum_position_to_matrix(positions[i]) for i in frame_indices],
                             dtype=np.float32)
        color_image_paths = [str(sequence_directory / associations.iloc[i, 1]) for i in frame_indices]
        depth_image_paths = [str(sequence_directory / associations.iloc[i, 3]) for i in frame_indices]
        color_images = np.array([cv2.imread(x).astype(np.float32) for x in color_image_paths])
        depth_images = np.array(
            [cv2.imread(x, cv2.IMREAD_UNCHANGED).astype(np.float32) / factor for x in depth_image_paths])
        camera_info = Camera(clip_depth_distance_threshold=clip_distance_threshold, camera_matrix=camera_matrix,
                             distance_koef=distance_koef, image_width=640, image_height=480)
        return camera_info, TUMDatasetLoader(camera_info, color_images, depth_images, positions)

    @staticmethod
    def tum_position_to_matrix(tum_position):
        """
        Convert TUM position format to matrix form.
        Transformation from world to camera frame.
        :param tum_position: [tx ty tz qx qy qz qw]
        :return:    [[R 0],
                     [T 1]]
        """
        matrix_form = np.eye(4)
        rotation = R.from_quat(tum_position[3:])
        matrix_form[:3, :3] = rotation.as_matrix().T
        matrix_form[3, :3] = tum_position[:3]
        assert np.all(matrix_form[:3, 3] == 0)
        assert matrix_form[3, 3] == 1
        return matrix_form
