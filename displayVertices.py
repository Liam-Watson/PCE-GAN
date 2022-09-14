'''
Simple scheme used for displaying point cloud data
'''
import os 
import sys
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import v
import time as t

fle = (sys.argv[1])
displayMode = int(sys.argv[2])

vertices = np.load(fle)

classes = {
    0: 'bareteeth',
    1: 'high_smile',
    2: 'cheeks_in',
    3: 'eyebrow',
    4: 'lips_back',
    5: 'lips_up',
    6: 'mouth_down',
    7: 'mouth_extreme',
    8: 'mouth_middle',
    9: 'mouth_up'
}
print(len(vertices))
vis = o3d.visualization.Visualizer()
vis.create_window()
count = 0
if displayMode == 0:
    for x in vertices:
            vis.clear_geometries()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(x)

            vis.add_geometry(pcd)

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            t.sleep(0.1)
            print(count)
            count+=1
elif displayMode == 1:
    for x in vertices:
        vis.clear_geometries()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        radii = [0.005, 0.01, 0.02, 0.04]
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        vis.add_geometry(rec_mesh)

        vis.update_geometry(rec_mesh)
        vis.poll_events()
        vis.update_renderer()

elif displayMode == 2:
    for x in vertices:
        vis.clear_geometries()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        
        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
        for alpha in np.logspace(np.log10(0.5), np.log10(0.001), num=20):
            vis.clear_geometries()
            print(f"alpha={alpha:.3f}")
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha, tetra_mesh, pt_map)
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()

            vis.add_geometry(mesh)

            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()

elif displayMode == 3:
    for x in vertices:
        vis.clear_geometries()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=12)
        densities = np.asarray(densities)
        density_colors = plt.get_cmap('plasma')(
            (densities - densities.min()) / (densities.max() - densities.min()))
        density_colors = density_colors[:, :3]
        density_mesh = o3d.geometry.TriangleMesh()
        density_mesh.vertices = mesh.vertices
        density_mesh.triangles = mesh.triangles
        density_mesh.triangle_normals = mesh.triangle_normals
        density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)

        vis.add_geometry(density_mesh)

        vis.update_geometry(density_mesh)
        vis.poll_events()
        vis.update_renderer()





# geometry is the point cloud used in your animaiton

