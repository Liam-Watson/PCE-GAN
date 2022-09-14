'''
Initial implementation of simple DCGAN for face generation
'''
import sys
from simpleDCGAN import Generator
import torch
import torch.nn as nn
import open3d as o3d
import numpy as np
import time
import matplotlib.pyplot as plt
from pointnet.model import PointNetCls
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random

# Hyperparameters 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("RUNNING ON: ", device)
NOISE_DIM = 128


modelPramsPath = sys.argv[1]
gen = Generator(NOISE_DIM).to(device)
gen.load_state_dict(torch.load(modelPramsPath))


randomSeed = random.randint(1, 10000)  # fix seed 7063 ( head on) #2167
randomSeed = 4670
print("Random Seed: ", randomSeed)
random.seed(randomSeed)
torch.manual_seed(randomSeed)



def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

vis = o3d.visualization.Visualizer()
vis.create_window()

def displayVerts(verts, numDisp=1, block=True, dispMethod=0):
    for vertices in verts[:numDisp]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)

        if dispMethod == 0:
            if block:
                o3d.visualization.draw_geometries([pcd])
            else:
                vis.clear_geometries()
                vis.add_geometry(pcd)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
        elif dispMethod == 1:
            pcd = pcd.uniform_down_sample(every_k_points=2)
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(100)
            
            tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
            for alpha in np.logspace(np.log10(0.006), np.log10(0.001), num=1):
                # vis.clear_geometries()
                # print(f"alpha={alpha:.3f}")
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha, tetra_mesh, pt_map)
                mesh.compute_vertex_normals()
                mesh.compute_triangle_normals()
                if block:
                    o3d.visualization.draw_geometries([mesh])
                else:
                    vis.clear_geometries()
                    vis.add_geometry(mesh)
                    vis.update_geometry(mesh)
                    vis.poll_events()
                    vis.update_renderer()
        elif dispMethod == 2:
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(100)
            with o3d.utility.VerbosityContextManager(
                    o3d.utility.VerbosityLevel.Debug) as cm:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=8, linear_fit = True, n_threads=16)
            densities = np.asarray(densities)
            density_colors = plt.get_cmap('plasma')(
                (densities - densities.min()) / (densities.max() - densities.min()))
            density_colors = density_colors[:, :3]
            density_mesh = o3d.geometry.TriangleMesh()
            density_mesh.vertices = mesh.vertices
            density_mesh.triangles = mesh.triangles
            density_mesh.triangle_normals = mesh.triangle_normals
            density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
            if block:
                o3d.visualization.draw_geometries([density_mesh])
            else:
                vis.clear_geometries()
                vis.add_geometry(density_mesh)
                vis.update_geometry(density_mesh)
                vis.poll_events()
                vis.update_renderer()
        elif dispMethod ==3:
            
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(300)
            radii = [0.005, 0.01, 0.02, 0.04]
            rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
            if block:
                o3d.visualization.draw_geometries([rec_mesh])
            else:
                vis.clear_geometries()
                vis.add_geometry(rec_mesh)
                vis.update_geometry(rec_mesh)
                vis.poll_events()
                vis.update_renderer()

z = torch.randn(4, NOISE_DIM, 1, device=device).requires_grad_()
print(z.shape)
classifier = PointNetCls(k=4, feature_transform=False)
classifier.load_state_dict(torch.load("./pointnet/utils/cls/cls_model_149.pth"))
classifier.to(device)
opt = torch.optim.Adam(classifier.parameters(), lr=0.00000001)
gen.to(device)
# lr=0.00001
lr = 0.1
expression_score = torch.tensor([-1000], device="cuda")
expProb = []
expProbFig = plt.figure(1)
expProbAx = expProbFig.add_subplot(1,1,1)
while np.exp(expression_score.to("cpu").detach().numpy()) < 0.9995:
    classifier.zero_grad()
    opt.zero_grad()
    fake = gen(z) # 1
    fake2 = fake.transpose(2, 1)
    expression_score = classifier(fake2)[0][:,0][0]
    expression_score.backward() # 3
    z.data = z + (z.grad*lr) # 4
    fake = (fake.to('cpu').double().detach().numpy())

    displayVerts(fake, numDisp=1, block=False, dispMethod=0)
    expProb.append(np.exp(expression_score.to("cpu").detach().numpy()))
    expProbAx.clear()
    expProbAx.plot(expProb, label="exp prob")
    expProbFig.canvas.draw_idle()
    expProbFig.canvas.flush_events()
    expProbFig.show()

        