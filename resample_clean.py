from sample_distribution_learn import SamplingDistributionFinder
import random
import collections
import math
from math import log
from scipy.stats.stats import pearsonr
import copy
import numpy as np


class ResampleClean:

    def __init__(self, data, sample_size):

        self.sample_size = sample_size
        self.data = data
        self.sample = SamplingDistributionFinder.sampling(data, self.sample_size)
        # print self.sample
        sample_dist = SamplingDistributionFinder.sample_distribution(self.sample)
        # print sample_dist
        self.acception_dist = SamplingDistributionFinder.resample_value_acception_distribution(sample_dist, None)
        # print self.acception_dist
        self.sample_should_have_dup = 0


    def resampling(self):
        """
        This method resample based on the learned sample
        :return:
        """
        tmpresam = SamplingDistributionFinder.sampling(self.data, self.sample_size)
        resample = []
        while len(resample) != self.sample_size:
            inclusion_map = [0] * len(tmpresam)
            for el in tmpresam:
                rand = random.random()
                acception = self._find_acc_prob(el, self.acception_dist)
                if rand < acception:
                    inclusion_map[tmpresam.index(el)] = 1
            for i in range(len(tmpresam)):
                if inclusion_map[i] == 1:
                    resample.append(tmpresam[i])
            tmp_resample_size = self.sample_size - len(resample)
            if tmp_resample_size != 0:
                tmpresam = SamplingDistributionFinder.sampling(self.data, tmp_resample_size)
        self.sample = resample
        sample_dist = SamplingDistributionFinder.sample_distribution(self.sample)
        self.acception_dist = SamplingDistributionFinder.resample_value_acception_distribution(sample_dist,
                                                                                           self.acception_dist)

    def distributed_sampler(self, accept_prob):
        """
        This method create sample with given accept_prob probability
        :param accept_prob:
        :return resample: list[]
        """
        tmpresam = SamplingDistributionFinder.sampling(self.data, self.sample_size)
        resample = []
        while len(resample) != self.sample_size:
            inclusion_map = [0] * len(tmpresam)
            for el in tmpresam:
                rand = random.random()
                acception = self._find_acc_prob(el, accept_prob)
                if rand < acception:
                    inclusion_map[tmpresam.index(el)] = 1
            for i in range(len(tmpresam)):
                if inclusion_map[i] == 1:
                    resample.append(tmpresam[i])
            tmp_resample_size = self.sample_size - len(resample)
            if tmp_resample_size != 0 :
                tmpresam = SamplingDistributionFinder.sampling(self.data, tmp_resample_size)
        return resample

    def _find_acc_prob(self, value, acception_dist):
        for elemnt in acception_dist:
            if elemnt[0] == value:
                return elemnt[1]
        return 1

    def is_repeated(self, sample):
        if len(list(set(sample))) == len(sample):
            return 0
        return 1

    def truth_sample(self):
        stop = 0
        # while self.is_repeated(self.sample):
        #     self.resampling()
        #     stop += 1
        # return stop
        # fx = open('dupropygain.csv', 'w')
        # threshold_tau = 1
        # while threshold_tau > 0.1:
        #     threshold_tau = self.dupropy_gain(SamplingDistributionFinder.sampling(self.data, 2 * self.sample_size),
        #                             self.acception_dist)
        #     # tmp = self.dupropy_gain(self.data, self.acception_dist)
        #     # fx.write(str(tmp) + ',\n')
        #     # print tmp
        #     self.resampling()
        #     # x = tmp
        #     stop += 1
        # # fx.close()


        # prev_dist = []
        # while len(list(set(self.acception_dist)-set(prev_dist))) != 0:
        #     prev_dist = copy.copy(self.acception_dist)
        #     self.resampling()
        #     stop += 1
        # return stop
        #

        # s = self.distributed_sampler(self.acception_dist)
        # while self.sam_entropy(s) < 1-(1.0/self.sample_size):
        #     prev_dist = copy.copy(self.acception_dist)
        #     self.resampling()
        #     stop += 1
        # return stop

        prev_dist = []
        while self.distribution_distance(prev_dist,self.acception_dist) > 0.001:
            prev_dist = copy.copy(self.acception_dist)
            self.resampling()
            stop += 1
        return stop


    def distribution_distance(self, prev_dist, current_dist):
        len_prev = len(prev_dist)
        len_current = len(current_dist)

        if len_current == 0 or len_prev == 0 or len_current > len_prev:
            return 1
        else:
            np_arr_cuurent = np.array(current_dist)
            np_arr_prev = np.array(prev_dist)
            dist = np.linalg.norm(np_arr_cuurent - np_arr_prev)
            return dist


    def dupropy(self, sample):
        dupropy_val = 0
        dup_values = [item for item, count in collections.Counter(sample).items() if count > 1]
        freq = []
        total_dup = 0
        for v in dup_values:
            f = [index for index in range(len(sample)) if sample[index] == v]
            total_dup += len(f)
            freq.append((v,len(f)))
        for fi in freq:
            if len(dup_values) > 1:
                en = -1.0*(float(fi[1])/total_dup)*math.log(float(fi[1])/total_dup,len(dup_values))
                dupropy_val += en
        return dupropy_val, dup_values, total_dup

    def dupropy_gain(self, sample, distribution):

        dist_pred = 0
        dupropy_val, dup_values, total_dup = self.dupropy(sample)
        freq_hat = []
        for p_hat in distribution:
            for dup in dup_values:
                if p_hat[0] == dup:
                    freq_hat.append(p_hat[1])
        partition = sum(freq_hat)
        p = [a/partition for a in freq_hat] # Normalize the probability
        for i in p :
            if len(p) > 1:
                dist_pred += -1.0*i*math.log(i,len(dup_values))
        return dupropy_val - dist_pred

    def dist_correlation(self, sample ):
        if len(self.acception_dist) >1:
            dist = SamplingDistributionFinder.sample_distribution(sample)
            sample_dist = SamplingDistributionFinder.resample_value_acception_distribution(dist, None)
            acc_val = []
            sam_acc = []
            for tup in self.acception_dist:
                acc_val.append(tup[0])
            for tup in sample_dist:
                sam_acc.append(tup[0])
            intersection = list(set(acc_val).intersection(set(sam_acc)))
            if len(intersection) > 2:
                list1 = []
                list2 = []
                for shared in intersection:
                    for d in sample_dist:
                        if d[0] == shared:
                            list1.append(d[1])
                    for d in self.acception_dist:
                        if d[0] == shared:
                            list2.append(d[1])
                corr = pearsonr(list1, list2)
                return corr
            else:
                return 0
        else:

            return 0

    def sam_entropy(self, sample):
        """
        This function calculate the sample entropy

        :return:
        """
        dist = SamplingDistributionFinder.sample_distribution(sample)
        ent = 0
        base = len(dist)
        sam_size = len(sample)
        if base > 1:
            for tup in dist:
                ent += -1.0*float(float(tup[1])/sam_size)*log(float(tup[1])/sam_size,base)
            return ent
        return 0






