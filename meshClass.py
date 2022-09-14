'''
Author: Liam Watson
Wrapper class for open3d point cloud object operations needed for PCE-GAN
'''
import sys
import torch
import torch.nn as nn
import open3d as o3d
import numpy as np

'''
Class for mesh operations
'''
class MeshClass():
    def __init__(self, points, visibility=True, h=800, w=1000):
        self.vis = o3d.visualization.Visualizer() # create a visualizer
        self.pcd = o3d.geometry.PointCloud() # create a point cloud
        self.vis.create_window(visible=visibility, height=h, width=w) # create a window
        if not points is None: # if points are given
            self.pcd.points = o3d.utility.Vector3dVector(points) # set the points
            
            
    # function for displaying point cloud
    def displayPointCloud(self, block=False):
        self.vis.create_window()
        if block:
            o3d.visualization.draw_geometries([self.pcd])
        else:
            
            self.vis.clear_geometries()
            self.vis.add_geometry(self.pcd)
            self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
    # function for setting point cloud - used for itterative generation
    def setPointCloud(self, pointCloud):    
        self.pcd.points = o3d.utility.Vector3dVector(pointCloud)

    # function for down sampling point cloud using k'th point method
    def downSampleUniform(self, every_k_points=2):
        self.pcd = self.pcd.uniform_down_sample(every_k_points=every_k_points)

    # function for downsamplingusing voxel based method
    def downSampleVoxel(self, voxel_size=0.01):
        self.pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)

    # function for estimating normals
    def estimateNormals(self, neighbors=100):
        self.pcd.estimate_normals()
        self.pcd.orient_normals_consistent_tangent_plane(neighbors)
    
    # Alpha mesh function
    def meshAlphaShape(self,alpha=0.006):
        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(self.pcd) 
        self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(self.pcd, alpha, tetra_mesh, pt_map)
        self.mesh.compute_vertex_normals()
        self.mesh.compute_triangle_normals()

    # poisson mesh function
    def meshPoisson(self, depth=7, scale=1.1, threads=16, crop=True):
        # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:

        mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.pcd, depth=depth,scale=scale, linear_fit = False, n_threads=threads)
        if crop:
            bbox = self.pcd.get_axis_aligned_bounding_box()
            self.mesh = mesh_poisson.crop(bbox)

    # Clean mesh 
    def cleanMesh(self):
        self.mesh.remove_degenerate_triangles()
        self.mesh.remove_duplicated_triangles()
        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_non_manifold_edges()

    # function for smoothing mesh
    def meshSimpleSmooth(self, iters=10):
        self.mesh = self.mesh.filter_smooth_simple(number_of_iterations=iters)
    # function for laplacian smoothing
    def meshLaplacianSmooth(self, iters=10):
        self.mesh = self.mesh.filter_smooth_laplacian(number_of_iterations=iters)
    # function for taubin smoothing
    def meshTaubinSmooth(self, iters=10):
        self.mesh = self.mesh.filter_smooth_taubin(number_of_iterations=iters)
    # function for generating a ball pivot mesh
    def meshBallPivot(self, radii=[0.005, 0.01, 0.02, 0.04]):
        distances = self.pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(self.pcd, o3d.utility.DoubleVector([radius, radius * 2]))   
    # function for displaying mesh
    def meshDisplay(self, block=False):
        # self.vis.create_window()
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color([0.5,0.5,0.5])
        if block:
            o3d.visualization.draw_geometries([self.mesh])  
        else:
            self.vis.clear_geometries()
            self.vis.add_geometry(self.mesh)
            self.vis.update_geometry(self.mesh)
            self.vis.poll_events()
            self.vis.update_renderer()
    # function to update visualisation with new events 
    def updateRenderer(self):
        self.vis.poll_events()
        self.vis.update_renderer()
    # function that employs outlier isolation to simplify high density points (improves surface reconstruction)
    def outlierStrat(self, radius=0.003, nb_points = 2, voxel_size = 1, every_k = 50):
        cl , ind = self.pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        inlier_cloud = self.pcd.select_by_index(ind)
        outlier_cloud = self.pcd.select_by_index(ind, invert=True)
        self.pcd = inlier_cloud
        self.downSampleVoxel(voxel_size=voxel_size)
        self.downSampleUniform(every_k_points=every_k)
        self.pcd.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(self.pcd.points), np.asarray(outlier_cloud.points)), axis=0))
        self.pcd.estimate_normals()
    # used for outlier display
    def display_inlier_outlier(self, cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud])

    def hiddenPointRemoval(self, radius):
        self.mesh, rem = self.pcd.hidden_point_removal([])
    # write point cloud to file
    def savePointCloud(self, filename):
        o3d.io.write_point_cloud(filename, self.pcd)
    # capture 2D raster of visualisation
    def capture_image(self):
        return np.asarray(self.vis.capture_screen_float_buffer())
    # destroy window object
    def destroy_window(self):
        self.vis.destroy_window()
    # rotate object
    def rotate(self, y = 90, x = 90, z = 90):
        R = self.pcd.get_rotation_matrix_from_xyz((x, y, z))
        self.pcd.rotate(R, center=(0, 0, 0))
        # rot = np.array([3,1, 1], dtype=np.float64)
        # self.pcd.rotate(rot)
    # load point cloud from file
    def loadMeshFile(self, filename):
        self.pcd = o3d.io.read_point_cloud(filename)

