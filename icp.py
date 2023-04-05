import os
import logging
import time

import open3d as o3d

from open3d.t.geometry import PointCloud


class ICP:

    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 voxel_sizes_list: list[float],
                 max_correspondence_distances_list: list[float],
                 criteria_param_list: list[list[float]],
                 use_gpu=True,
                 gpu_id=0,):

        self.input_files = os.listdir(input_dir)
        self.input_files.sort()
        self.output_dir = output_dir

        self.gpu_id = gpu_id

        self.voxel_sizes_list = voxel_sizes_list
        self.max_correspondence_distances_list = max_correspondence_distances_list
        self.criteria_param_list = criteria_param_list

        cuda_available = o3d.core.cuda.is_available()
        if cuda_available:
            logging.info("CUDA is available")
            if use_gpu:
                logging.info("Using GPU acceleration")
                self.device = o3d.core.Device(f"cuda:{gpu_id}")
                self.is_gpu = True
            else:
                logging.info("Force not to use GPU acceleration")
                self.device = o3d.core.Device("cpu:0")
                self.is_gpu = False
        else:
            logging.info("CUDA is not available")
            self.device = o3d.core.Device("cpu:0")
            self.is_gpu = False

        self.tensor = o3d.core.Tensor([], dtype=o3d.core.Dtype.Float32, device=self.device)

        logging.info(f"Using device: {self.device}")

    @staticmethod
    def draw_registration_result(source, target, transformation):
        source_temp = source.clone()
        target_temp = target.clone()

        source_temp.transform(transformation)

        # This is patched version for tutorial rendering.
        # Use `draw` function for you application.
        o3d.visualization.draw_geometries(
            [source_temp.to_legacy(),
             target_temp.to_legacy()])

    @staticmethod
    def iteration_callback(loss_log_map):
        logging.info(f"Iteration Index: {loss_log_map['iteration_index'].item():3}, "
                     f"Scale Index: {loss_log_map['scale_index'].item():3}, "
                     f"Scale Iteration Index: {loss_log_map['scale_iteration_index'].item():3}, "
                     f"Fitness: {loss_log_map['fitness'].item():.15f}, "
                     f"Inlier RMSE: {loss_log_map['inlier_rmse'].item():.15f}")

    def pair_align(self,
                   source: PointCloud,
                   target: PointCloud):

        if self.is_gpu:
            # noinspection PyUnresolvedReferences
            import open3d.cuda.pybind.t.pipelines.registration as treg
        else:
            # noinspection PyUnresolvedReferences
            import open3d.cpu.pybind.t.pipelines.registration as treg

        icp_source = source
        icp_target = target
        if self.is_gpu:
            icp_source = source.cuda(self.gpu_id)
            icp_target = target.cuda(self.gpu_id)

        criteria_list = []
        for p in self.criteria_param_list:
            criteria_list.append(treg.ICPConvergenceCriteria(p[0], p[1], p[2]))

        voxel_sizes = o3d.utility.DoubleVector(self.voxel_sizes_list)
        max_correspondence_distances = o3d.utility.DoubleVector(self.max_correspondence_distances_list)

        init_source_to_target = self.tensor.eye(4, o3d.core.Dtype.Float32)
        estimation = treg.TransformationEstimationPointToPoint()

        s = time.time()

        registration_ms_icp: o3d.t.pipelines.registration.RegistrationResult = \
            treg.multi_scale_icp(icp_source, icp_target, voxel_sizes,
                                 criteria_list,
                                 max_correspondence_distances,
                                 init_source_to_target, estimation,
                                 lambda x: self.iteration_callback(x))

        ms_icp_time = time.time() - s
        logging.info(f"Time taken by Multi-Scale ICP: {ms_icp_time}")
        logging.info(f"Inlier Fitness: {registration_ms_icp.fitness}")
        logging.info(f"Inlier RMSE: {registration_ms_icp.inlier_rmse}")
        logging.info(f"Transformation Matrix: \n{registration_ms_icp.transformation}")

        with open(os.path.join(self.output_dir, "transformation_matrix.txt"), "a") as f:
            f.write(registration_ms_icp.transformation)

        self.draw_registration_result(icp_source, icp_target, registration_ms_icp.transformation)

    def align(self):
        input_point_clouds: list[PointCloud] = []
        for file in self.input_files:
            input_point_clouds.append(o3d.t.io.read_point_cloud(file))

        for pc in input_point_clouds:
            if self.is_gpu:
                pc.cuda(self.gpu_id)
            pc.estimate_normals(30, 0.1)

        for i in range(len(input_point_clouds) - 1):
            self.pair_align(input_point_clouds[i], input_point_clouds[i + 1])
