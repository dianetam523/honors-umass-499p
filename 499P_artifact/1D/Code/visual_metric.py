import dpcomp_core.cartesian_product as cp
from dpcomp_core.execution import assemble_experiments
from dpcomp_core.execution import ListWriter
from dpcomp_core.execution import process_experiments
from dpcomp_core import dataset
from dpcomp_core import workload
from dpcomp_core.algorithm import *
from dpcomp_core import util

#####################################################################
# This class calculates different types of error metrics for comparing 
# differentially private noisy data and original data.
#####################################################################

class VisualMetric():
    def __init__(self, orig_data, noisy_data, settings):
        self.orig_data = orig_data.tolist()
        self.noisy_data = noisy_data
        self.maxNoisy = max(self.noisy_data)
        self.maxNoisyIndex = self.noisy_data.index(max(self.noisy_data))
        self.minNoisy = min(self.noisy_data)
        self.minNoisyIndex = self.noisy_data.index(min(self.noisy_data))
        self.maxOrig = max(self.orig_data)
        self.maxOrigIndex = self.orig_data.index(max(self.orig_data))
        self.maxOrig = max(self.orig_data)
        self.maxOrigIndex = self.orig_data.index(max(self.orig_data))
        self.settings = settings
        self.scale = settings[6]
        self.domain = settings[7]
    
    # Calculates and returns the score for the specified error metric 
    # given as parameter "met" 
    def calcError(self, met):
        score = 0
        Lmet = getLmetrics(self.settings)
        # L1-norm error
        if met == "L1":
            score = Lmet[0]
            
        # L2-norm error
        elif met == "L2":
            score = Lmet[1]
            
        # L2-norm error
        elif met == "Linf":
            score = Lmet[2]
          
        elif met == "chi-sq":
            for n in range(len(self.orig_data)):
                num = float(self.orig_data[n] - self.noisy_data[n])*float(self.orig_data[n] - self.noisy_data[n])
                denom = float(self.orig_data[n] + self.noisy_data[n])
                if denom > 0:
                    score += num/denom
                else:
                    score += 0
            
        # Difference of true vs noisy max INDEX
        elif met == "M1":
            score = abs(self.maxNoisyIndex-self.maxOrigIndex)
        
        # Difference of true vs noisy max VALUE
        elif met == "M2":
            score = abs(self.maxNoisy-self.maxOrig)
        
        # Difference of true vs noisy 2nd max INDEX
        elif met == "M3":
            orig_data_copy = self.orig_data[:]
            orig_data_copy.remove(self.maxOrig)
            
            noisy_data_copy = self.noisy_data[:]
            noisy_data_copy.remove(self.maxNoisy)

            second_maxOrig = max(orig_data_copy)
            second_maxOrigIndex = orig_data_copy.index(max(orig_data_copy))
            
            second_maxNoisy = max(noisy_data_copy)
            second_maxNoisyIndex = noisy_data_copy.index(max(noisy_data_copy))
            
            score = abs(self.maxNoisy-second_maxOrigIndex)
        else:
            raise ValueError("Please specify a valid error metric.")            
        return score

    
# Helper methods
def getLmetrics(params):
    #unpack settings params
    eps, curr_ex_s, ds_s, alg, ds_nickname, wl, sample, domain = params

    # common parameters
    query_sizes = [2000]
    # privacy level epsilon
    epsilons = [eps]
    # experiment seeds
    ex_seeds = [curr_ex_s]
    # dataset sampling seeds, used to generate different data instances based on the same dataset configuration.
    ds_seeds = [ds_s]
    # workload sampling seeds, used to generate different workload instances for workloads that take a seed parameter.
    w_seeds = [0]


    # 1D algorithm specification
    d1_algorithms = [alg]
    # 1D dataset specification
    d1_dataset_map = [(dataset.DatasetSampledFromFile, {'nickname': ds_nickname})]
    # 1D workload specification
    d1_workload_map = [(wl, {})]
    # 1D scales
    d1_scales = [sample]
    # 1D domains
    d1_domains = [domain]

    # create cartesian product of all 1D parameter values
    d1_params_map = assemble_experiments(d1_algorithms,
                                         d1_dataset_map,
                                         d1_workload_map,
                                         d1_domains,
                                         epsilons,
                                         d1_scales,
                                         query_sizes,
                                         ex_seeds,
                                         ds_seeds,
                                         w_seeds)


    # combine 1D and 2D parameters
    params_map = dict(d1_params_map.items())


    # run experiments
    writer = ListWriter()
    process_experiments(params_map, writer)
    writer.close()
    # collect sample errors
    for group in writer.metric_groups:
        for sampleError in group:
#             print sampleError.E.A.short_name
#             print sampleError.E.X.fname
#             print sampleError.W
#             print sampleError.error_payload
            sampleL1 = sampleError.error_payload["TypeI.L1"]
            sampleL2 = sampleError.error_payload["TypeI.L2"]
            sampleLinf = sampleError.error_payload["TypeI.Linf"]
    
    return sampleL1, sampleL2, sampleLinf