import numpy as np
import torch
import os 
import sys
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import math

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
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

def sort_faces(faces):
    num = 0
    new_arr = np.array([face_sort(faces[0]), face_sort(faces[1])])
    l = len(faces)
    printProgressBar(0, l-2, prefix = 'Sorting:', suffix = 'Complete', length = 50)
    for i in faces[2::]:
        printProgressBar(num + 1, l-2, prefix = 'Sorting:', suffix = 'Complete', length = 50)
        num += 1
        new_arr = np.append(new_arr, np.array([face_sort(i)]), 0)
    return new_arr

def face_sort(face):
    sorted_face = None
    if face[1][0] > face[0][0]:
        sorted_face = np.array([face[0], face[1]])
    else:
        sorted_face = np.array([face[1], face[0]])
    for v in face[2::]:
        i = 0
        while (v[0] + v[1] + v[2]) >= sorted_face[i][0] + sorted_face[i][1] + sorted_face[i][2]:
            if i >= len(sorted_face)-1:
                break
            else:
                i += 1

        if i >= len(sorted_face)-1:
            sorted_face = np.append(sorted_face, np.array([v]), 0)
        else:
            sorted_face = np.insert(sorted_face, np.array([i]), np.array([v]), 0)


    
    return sorted_face

def dist(v1, v2):
    return math.sqrt(math.pow(v1[0]- v2[0], 2) + math.pow(v1[1]- v2[1], 2) + math.pow(v1[2]- v2[2], 2))

def cluster_face(face):
        
    c_face = np.array([[0,0,0]])
    min_x = 10
    min_vert = None
    i = 0
    min_index = 0
    # print('\tfinding root...')
    for v in face:
        if v[1] < min_x:
            min_vert = v
            min_x = v[1]
            min_index = i
        i += 1
    # print(face.shape)
    face = np.delete(face, min_index, 0)
    c_face = np.append(c_face, [min_vert], 0)
    # print(face.shape)

    close = None
    num = 0
    # print('\tbuilding clusters...')
    for j in range(71):
        num+=1
        for k in range(71):
            x, y = face.shape
            if x == 1:
                c_face = np.append(c_face, [face[0]], 0)
                c_face = np.delete(c_face, 0, 0)
                break
            else:

            
                close = min_vert
                current = min_vert
                d = 10
                
                i = 0
                min_index = 0
                for v in face:
                    if dist(close, v) < d:
                        d = dist(close, v)
                        current = v
                        min_index = i
                    i += 1
                # print(current)
                face = np.delete(face, min_index, 0)
                c_face = np.append(c_face, [current], 0)
        min_vert = close

    return c_face

def cluster(faces):
    print('clustering')
    num = 0
    l = len(faces)
    printProgressBar(0, l, prefix = 'Clustering:', suffix = 'Complete', length = 50)
    new_arr = np.array([cluster_face(faces[0]), cluster_face(faces[1])])
    num += 2
    printProgressBar(num, l, prefix = 'Clustering:', suffix = 'Complete', length = 50)
    for i in faces[2::]:
        
        num += 1
        new_arr = np.append(new_arr, np.array([cluster_face(i)]), 0)
        printProgressBar(num, l, prefix = 'Clustering:', suffix = 'Complete', length = 50)
    return new_arr


kid_test = np.load("processedData/high_smile/test.npy")
kid_train = np.load("processedData/high_smile/train.npy")
print(kid_test.shape)
x, y, c = kid_test.shape
# kid_test = kid_test.reshape(x, c, y, 1)
# kid_test = np.array(torch.randn(1878, 3, 5023, 1))
kid_test = np.array(torch.randn(1878, 5023, 3))
x, y, c = kid_train.shape
# kid_train = kid_train.reshape(x, c, y, 1)

# print(kid_test.shape)
# print(kid_train.shape)
# print("data")
# kid_train = kid_train[0:100]
# print(kid_train)
# print('----------------------------------')
# # kid_train = np.sort(kid_train, 2)
# kid_train = sort_faces(kid_train) 
# print(kid_train)

vertices = cluster(kid_train[:5])
vis = o3d.visualization.Visualizer()
vis.create_window()

x = vertices[0]
print(x.shape)
# x = cluster_face(x)
print(x.shape)
print(x)

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

vis.run()

# np.save("quan_assessment_framework/metrics_test_rnd.npy", kid_test[0:200])
# np.save("quan_assessment_framework/metrics_test_true.npy", kid_train[0:300])

