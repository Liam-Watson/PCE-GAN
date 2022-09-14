'''
Class used for displaying a list of point cloud faces
'''
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time

class view_face:
    def __init__(self, faces):
        self.faces = faces
        self.vis = o3d.visualization.Visualizer()
        

    def display(self):
        self.vis.create_window()

       
        v = self.faces
        for x in v:
            self.vis.clear_geometries()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(x)
            self.vis.add_geometry(pcd)

            self.vis.update_geometry(pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(1)

        # self.vis.run()
        
        self.vis.destroy_window()
