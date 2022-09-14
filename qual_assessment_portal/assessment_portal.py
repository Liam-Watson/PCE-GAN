'''
Wrapper file to initialise qual_assessment_portal
'''
import numpy as np
from qual_assessment_portal.view_face import *

class assessment_portal:
    def __init__(self, gen_samples, test_samples):
        self.g = gen_samples
        self.t = test_samples

    def run_portal(self):
        np.random.shuffle(self.g)
        v = view_face(self.g)
        v.display()

if __name__ == '__main__':
    a = assessment_portal(np.load('qual_assessment_portal/metrics_test_rnd.npy'), np.load('qual_assessment_portal/metrics_test_true.npy'))
    a.run_portal()