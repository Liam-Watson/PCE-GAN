import numpy as np
import torch
import os 
import sys
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

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
    new_arr = np.array([face_sort(faces[0])[0:5024]])
    l = len(faces)
    printProgressBar(0, l-2, prefix = 'Sorting:', suffix = 'Complete', length = 50)
    for i in faces[1::]:
        printProgressBar(num + 1, l-2, prefix = 'Sorting:', suffix = 'Complete', length = 50)
        num += 1
        new_arr = np.append(new_arr, np.array([face_sort(i)[0:5024]]), 0)
    return new_arr

def face_sort(face):
    sorted_face = None
    if (face[1][0] + face[1][1] + face[1][2]) > (face[0][0] + face[0][1]+ face[0][2]):
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



kid_test = np.load("processedData/high_smile/test.npy")
kid_train = np.load("processedData/high_smile/train.npy")
x, y, c = kid_test.shape
# kid_test = kid_test.reshape(x, c, y, 1)
# kid_test = np.array(torch.randn(1878, 3, 5023, 1))
kid_test = np.array(torch.randn(1878, 5023, 3))
# x, y, c = kid_train.shape
# kid_train = kid_train.reshape(x, c, y, 1)
kid_test = kid_test[0:200]
kid_train = kid_train[0:300]

# kid_test = sort_faces(kid_test)
# kid_train = sort_faces(kid_train)



np.save("quan_assessment_framework/metrics_test_rnd.npy", kid_test)
np.save("quan_assessment_framework/metrics_test_true.npy", kid_train)
np.save("qual_assessment_portal/metrics_test_rnd.npy", kid_test)
np.save("qual_assessment_portal/metrics_test_true.npy", kid_train)

