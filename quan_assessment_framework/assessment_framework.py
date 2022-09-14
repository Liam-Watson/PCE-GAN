from quan_assessment_framework.calculate_metrics import metrics
import numpy as np

class q_framework:
    def __init__(self, generated_samples, testing_samples):
        self.g_samples = generated_samples
        self.t_samples = testing_samples
        self.calc_metrics = metrics()
        self.KID = None
        self.FID = None
        self.specificities = []
        self.generalisations = []
        

    def calc_s(self):
        status = 0
        l = len(self.g_samples)
        self.printProgressBar(0, l, prefix = 'Specificity:', suffix = 'Complete', length = 50)
        for i in self.g_samples:
            self.printProgressBar(status + 1, l, prefix = 'Specificity:', suffix = 'Complete', length = 50)
            status += 1
            self.update_specificity(self.calc_metrics.calculate_specificity(i, self.t_samples))

    def calc_g(self):
        status = 0
        l = len(self.t_samples)
        self.printProgressBar(0, l, prefix = 'Generalisation:', suffix = 'Complete', length = 50)
        for i in self.t_samples:
            self.printProgressBar(status + 1, l, prefix = 'Generalisation:', suffix = 'Complete', length = 50) 
            status += 1
            self.update_generalisation(self.calc_metrics.calculate_generalisation(self.g_samples, i))

    def calc_fid_kid(self):
        print("calcualting KID")
        results_f, results_k = self.calc_metrics.calculate_KID(self.g_samples, self.t_samples)
        self.update_KID(results_k, results_f)
        return results_f, results_k


    def update_KID(self, results_kid, results_fid):
        self.KID = results_kid
        self.FID = results_fid

        

    def update_generalisation(self, new_generalisation):
        # np.append(self.generalisations,new_generalisation)
        self.generalisations.append(new_generalisation)

    def update_specificity(self, new_specificity):
        # np.append(self.specificities, new_specificity)
        self.specificities.append(new_specificity)

    def get_specificity(self):
        return 100*(sum(self.specificities)/len(self.specificities)), 100*(np.std(self.specificities))
    
    def get_generalisation(self):
        
        return 100*(sum(self.generalisations)/len(self.generalisations)), 100*(np.std(self.generalisations))

    def get_KID(self):
        
        for p, m, s in self.KID:
            print('KID: %.2f ± %.3f' % (m*10, s*10))

        for p, m, s in self.FID:
            print('FID: %.2f ± %.3f' % (m/10, s/10))

    def run_framework(self):
        self.calc_s()

        self.calc_g()
        print("GEN",self.get_generalisation())
        print("GEN",self.get_specificity())
        self.calc_fid_kid()

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


if __name__ == '__main__':
    f = q_framework(np.load("quan_assessment_framework/metrics_test_rnd.npy"), np.load("quan_assessment_framework/metrics_test_true.npy"))
    f.run_framework()
    g, g_std = f.get_generalisation()
    print('Generalisation:',g, '±',g_std)
    s, s_std = f.get_specificity()
    print('Specificity',s, '±', s_std)
    f.get_KID()