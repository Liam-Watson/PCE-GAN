import math
import sys
import torch
import numpy as np
from quan_assessment_framework.ganmetrics.kid_score import calculate_kid_given_paths
from quan_assessment_framework.ganmetrics.fid_score import calculate_fid_given_paths
import open3d as o3d
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meshClass import MeshClass

class metrics():
    def __init__(self):
        pass

    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar - found at https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters?noredirect=1&lq=1
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()

    def pc_to_img_arr(self, data_set):
        l = len(data_set)
        self.printProgressBar(0, l, prefix = 'Rasterizing samples:', suffix = 'Complete', length = 50)
        
        m = MeshClass(None, visibility=True, h=64, w=64)

        # vis = o3d.visualization.Visualizer()
        # vis.create_window(visible=False) #works for me with False, on some systems needs to be true

        i1 = None
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(data_set[0])

        # pcd.estimate_normals()
        # pcd.orient_normals_consistent_tangent_plane(100)
        # radii = [0.005, 0.01, 0.02, 0.04]
        # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

        # vis.add_geometry(rec_mesh)
        # vis.update_geometry(rec_mesh)
        # vis.poll_events()
        # vis.update_renderer()
        # i1 = np.asarray(vis.capture_screen_float_buffer())
        m.updateRenderer()
        m.setPointCloud(data_set[0])

        m.estimateNormals(neighbors=30)
        m.meshPoisson(depth=7, threads=16)
        # meshClass.meshBallPivot()
        m.meshLaplacianSmooth(iters=4)
        m.cleanMesh()
        m.meshDisplay(block=False)
        i1 = m.capture_image()

        self.printProgressBar(1, l, prefix = 'Rasterizing samples:', suffix = 'Complete', length = 50) 

        i2 = None
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(data_set[1])

        # pcd.estimate_normals()
        # pcd.orient_normals_consistent_tangent_plane(100)
        # radii = [0.005, 0.01, 0.02, 0.04]
        # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

        # vis.add_geometry(rec_mesh)
        # vis.update_geometry(rec_mesh)
        # vis.poll_events()
        # vis.update_renderer()
        # i2 = np.asarray(vis.capture_screen_float_buffer())
        m.updateRenderer()
        m.setPointCloud(data_set[1])

        m.estimateNormals(neighbors=8)
        m.meshPoisson(depth=7, threads=8)
        # meshClass.meshBallPivot()
        m.meshLaplacianSmooth(iters=4)
        m.cleanMesh()
        m.meshDisplay(block=False)
        i2 = m.capture_image()
        self.printProgressBar(2, l, prefix = 'Rasterizing samples:', suffix = 'Complete', length = 50) 

        out = [i1, i2]

        i = 0
        for x in data_set[2::]:
            i += 1
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(m)

            # pcd.estimate_normals()
            # pcd.orient_normals_consistent_tangent_plane(100)
            # radii = [0.005, 0.01, 0.02, 0.04]
            # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

            # vis.add_geometry(rec_mesh)
            # vis.update_geometry(rec_mesh)
            # vis.poll_events()
            # vis.update_renderer()
            # path = './quan_assessment_framework/images/' + str(i) + '.png'
            # vis.capture_screen_image(path)
            m.updateRenderer()
            m.setPointCloud(x)

            m.estimateNormals(neighbors=8)
            m.meshPoisson(depth=7, threads=8)
            # meshClass.meshBallPivot()
            m.meshLaplacianSmooth(iters=4)
            m.cleanMesh()
            m.meshDisplay(block=False)
            
            out.append(m.capture_image())
            self.printProgressBar(i + 2, l, prefix = 'Rasterizing samples:', suffix = 'Complete', length = 50) 
        m.destroy_window()

        # print(out)
        # print(np.array(out).shape)
        return np.array(out)



    def cluster_face(self, face):
        
        c_face = np.array([[0,0,0]])
        min_x = 10
        min_vert = None
        i = 0
        min_index = 0
        print('\tfinding root...')
        for v in face:
            if v[1] < min_x:
                min_vert = v
                min_x = v[1]
                min_index = i
            i += 1
        np.delete(face, min_index, 0)
        np.append(c_face, [min_vert], 0)

        close = None
        num = 0
        print('\tbuilding clusters...')
        for j in range(70):
            print('\t\tbuilding cluster',num)
            num+=1
            for i in range(70):
                
                close = min_vert
                current = min_vert
                d = 10
                
                i = 0
                min_index = 0
                for v in face:
                    if self.dist(close, v) < d:
                        d = self.dist(close, v)
                        current = v
                        min_index = i
                    i += 1

                np.delete(face, min_index, 0)
                np.append(c_face, [current], 0)
            min_vert = close

        return c_face


        
        
        

    def calculate_KID(self, generated_samples, testing_samples):
        
        

        # self.pc_to_img_arr(generated_samples[0:30])
        # x, y, c = generated_samples.shape
        # g = generated_samples.reshape(x, c, y, 1)
        # x, y, c = testing_samples.shape
        # t = testing_samples.reshape(x, c, y, 1)
        g = self.pc_to_img_arr(generated_samples)
        n, x, y, c = g.shape
        g = g.reshape(n, c, x, y)
        t = self.pc_to_img_arr(testing_samples)
        n, x, y, c = t.shape
        t = t.reshape(n, c, x, y)
        # t = np.array(torch.randn(n, c, x, y))

        path_t = 'quan_assessment_framework/ganmetrics/t.npy'
        path_g = 'quan_assessment_framework/ganmetrics/g.npy'
        np.save(path_t, t)
        np.save(path_g, g)

        results_fid = calculate_fid_given_paths([path_g]+[path_t], 10, '', 2048, model_type='inception')
        results_kid = calculate_kid_given_paths([path_g]+[path_t], 10, '', 2048, model_type='inception')
        return results_fid, results_kid
        

    def calculate_generalisation(self, generated_samples, test_sample):
        g = sys.float_info.max
        z = None
        for i in generated_samples:
            dist = self.distance(test_sample, i)
            if dist < g:
                z = i
        
        return self.distance(test_sample, z)

    # for the sample, the face closest to it from the testing samples and obtain the average per vertex distance
    def calculate_specificity(self, generated_sample, testing_samples):
        s = sys.float_info.max
        status = 0
        for i in testing_samples:
            # print("\t\t comparison -",status)
            # status += 1
            s = min(s, self.distance(generated_sample, i))

        return s

    def distance(self, face1, face2):
        n = min(len(face1), len(face2))
        sum = 0

        for i in range(n):
            sum += math.sqrt(math.pow(face1[i][0]-face2[i][0], 2) + math.pow(face1[i][1]-face2[i][1], 2) + math.pow(face1[i][2]-face2[i][2], 2))

        return sum/n

    def dist(self, v1, v2):
        return math.sqrt(math.pow(v1[0]- v2[0], 2) + math.pow(v1[1]- v2[1], 2) + math.pow(v1[2]- v2[2], 2))
