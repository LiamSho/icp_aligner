import logging

from icp import ICP


def main():
    # setup logger
    logging.basicConfig(level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        format='[%(asctime)s] [%(levelname)s] %(message)s')

    voxel_sizes_list = [0.1, 0.05, 0.025]
    max_correspondence_distances_list = [0.3, 0.14, 0.07]
    criteria_param_list = [
        [0.0001, 0.0001, 20],
        [0.00001, 0.00001, 15],
        [0.000001, 0.000001, 10]]

    icp = ICP(input_dir="input",
              output_dir="output",
              voxel_sizes_list=voxel_sizes_list,
              max_correspondence_distances_list=max_correspondence_distances_list,
              criteria_param_list=criteria_param_list,
              use_gpu=True,
              gpu_id=0)

    icp.align()


if __name__ == '__main__':
    main()
