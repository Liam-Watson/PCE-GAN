import numpy as np
import torch
import os 
import sys
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

kid_train = np.load("processedData/high_smile/train.npy")[0:10]
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False) #works for me with False, on some systems needs to be true
i = 0
for m in kid_train:
    i += 1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(m)

    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)
    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

    vis.add_geometry(rec_mesh)
    vis.update_geometry(rec_mesh)
    vis.poll_events()
    vis.update_renderer()
    path = './quan_assessment_framework/images/' + str(i) + '.png'
    # vis.capture_screen_image(path)
    print(np.asarray(vis.capture_screen_float_buffer()).shape)
vis.destroy_window()