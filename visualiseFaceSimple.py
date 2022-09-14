'''
Simple mesh and point cloud visualization class
'''
from meshClass import MeshClass
import numpy as np
import pymeshlab as pm
cls = MeshClass(None)


cls.loadMeshFile()

cls.displayPointCloud(block=True)

cls.estimateNormals(30)
cls.meshPoisson()
cls.meshTaubinSmooth(2)
cls.cleanMesh()
cls.meshDisplay(block=True)


