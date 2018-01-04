import random
import numpy as np
from scipy import stats
import math


class SamplingDistributionFinder:
    def __init__(self):
        pass

    @staticmethod
    def sampling(data, sample_size):
        """
        This method return a uniform sample from data
        :param sample_size:
        :return: sample list
        """

        sample = random.sample(data, sample_size)
        return sample

    @staticmethod
    def sample_distribution(sample):
        """
        This method return the frequency of values in the sample
        :param sample:
        :return: list [(value,frequency)]
        """
        sample_values = list(set(sample))
        frequency = [0] * len(sample_values)

        for element in sample:
            frequency[sample_values.index(element)] += 1
        return zip(sample_values, frequency)

    @staticmethod
    def value_probability(distribution):
        """
        This function calculate the experimental probablity
        :param distribution:
        :return: list probability
        """
        freq = []
        values = []
        for element in distribution:
            freq.append(element[1])
            values.append(element[0])
        data_size = sum(freq)
        probability = []
        for element in freq:
            probability.append(element / data_size)
        return probability

    @staticmethod
    def resample_value_acception_distribution(distribution, prev_acception_prob=None):
        """
        This method calculate the (value , accept probablility)
        :param data:
        :param distribution:
        :param prev_acception_prob:
        :return: list accept value
        """
        freq = []
        values = []
        for element in distribution:
            freq.append(element[1])
            values.append(element[0])

        # sample_size = sum(freq)
        # data_size = len(data)

        if prev_acception_prob is None:
            new_acception_prob = []
            for el in distribution:
                # x = float(sample_size) / float(el[1] * data_size)
                x = 1 / float(el[1])
                if x != 1:
                    new_acception_prob.append((el[0], x))
        else:
            new_acception_prob = prev_acception_prob
            for el in distribution:
                ind = -1
                for j in range(len(prev_acception_prob)):
                    if prev_acception_prob[j][0] == el[0]:
                        ind = j
                # x = float(sample_size) / float(el[1] * data_size)
                x = 1 / float(el[1])
                if ind == -1:
                    if x != 1:
                        new_acception_prob.append((el[0], x))
                else:
                    # x = float(sample_size) / float(el[1] * data_size)
                    x = 1 / float(el[1])
                    new_acception_prob[ind] = (el[0], new_acception_prob[ind][1] * x)
        return new_acception_prob

    def linear_program(self,sample):

        mean = np.mean(list(set(sample)))

        dist = SamplingDistributionFinder.sample_distribution(sample)
        val = []
        freq = []
        for el in dist:
            val.append(el[0])
            freq.append(el[1])
        coeff = []
        for v in val:
            coeff.append(v-mean)
        all_other_is_one_coeff_for_keep = []
        for keep in range(len(freq)):
            tmp = 0
            for j in range(len(freq)):
                if j != keep:
                    tmp += freq[j]*coeff[j]
            coeff_keep = float(tmp)/coeff[keep]
            if coeff_keep > 0:
                all_other_is_one_coeff_for_keep.append([val[keep],coeff_keep])
        return all_other_is_one_coeff_for_keep,val,freq,dist


    def linear_distribution(self, sample, prev_acception_prob = None):
        """
        This method calculate the (value , accept probablility)
        :param data:
        :param distribution:
        :param prev_acception_prob:
        :return: list accept value
        """
        all_other_is_one_coeff_for_keep, values, freq, distribution = self.linear_program(sample)

        if prev_acception_prob == None:
            new_acception_prob = []
            for el in distribution:
                # x = float(sample_size) / float(el[1] * data_size)
                x = 1 / float(el[1])
                if x != 1:
                    new_acception_prob.append((el[0], x))
        else:
            new_acception_prob = prev_acception_prob
            for el in distribution:
                ind = -1
                for j in range(len(prev_acception_prob)):
                    if prev_acception_prob[j][0] == el[0]:
                        ind = j
                # x = float(sample_size) / float(el[1] * data_size)
                x = 1 / float(el[1] )
                if ind == -1:
                    if x != 1:
                        new_acception_prob.append((el[0], x))
                else:
                    # x = float(sample_size) / float(el[1] * data_size)
                    x = 1 / float(el[1])
                    new_acception_prob[ind] = (el[0], new_acception_prob[ind][1] * x)
        return new_acception_prob

    @staticmethod
    def resample_dist_calc(distribution, novelty, prev_acception_prob = None):
        """
        This method calculate the (value , accept probablility)
        :param distribution:
        :param novelty:
        :param prev_acception_prob:
        :return:
        """
        freq = []
        values = []
        for element in distribution:
            freq.append(element[1])
            values.append(element[0])
        # sample_size = sum(freq)
        # data_size = len(data)
        if prev_acception_prob == None:
            new_acception_prob = []
            for el in distribution:
                # x = float(sample_size) / float(el[1] * data_size)
                x = 1 / float(el[1])
                if x != 1:
                    new_acception_prob.append((el[0], x))
        else:
            new_acception_prob = prev_acception_prob
            for el in distribution:
                ind = -1
                for j in range(len(prev_acception_prob)):
                    if prev_acception_prob[j][0] == el[0]:
                        ind = j
                # x = float(sample_size) / float(el[1] * data_size)
                x = 1 / float(el[1] )
                if ind == -1:
                    if x != 1:
                        new_acception_prob.append((el[0], x))
                else:
                    # x = float(sample_size) / float(el[1] * data_size)
                    x = 1 / float(el[1])
                    new_acception_prob[ind] = (el[0], new_acception_prob[ind][1] * x)
        summation = 0
        for p in new_acception_prob:
            summation += p[1]
        normalized = []
        for p in new_acception_prob:
            normalized.append([p[0],(p[1]/summation)*(1.0-novelty)])
        return normalized

    #######

    @staticmethod
    def accept_distribution_update(distribution, sample_size, virtual_dataset_size, confidence_value, num_of_resampling, prev_acception_prob=None):
        """
        This method calculate the (value , accept probablility)
        :param distribution:
        :param confidence_value:
        :param virtual_dataset_size:
        :param prev_acception_prob:
        :return: new acception prob, new virtual dataset size
        """

        # freq = []
        # values = []
        # for element in distribution:
        #     freq.append(element[1])
        #     values.append(element[0])
        curr_vir_size = virtual_dataset_size
        if prev_acception_prob == None:
            new_acception_prob = []

            for el in distribution:
                # x = float(sample_size) / float(el[1] * data_size)
                # x = 1 / float(el[1])
                if float(el[1]) != 1:
                    x = sample_size / (curr_vir_size * float(el[1]))
                    new_acception_prob.append((el[0], x))
                    virtual_dataset_size = virtual_dataset_size - (1.0/float(x)) + 1
        else:
            new_acception_prob = prev_acception_prob
            for el in distribution:
                ind = -1
                for j in range(len(prev_acception_prob)):
                    if prev_acception_prob[j][0] == el[0]:
                        ind = j
                x = float(sample_size) / float(el[1] * virtual_dataset_size)
                if float(el[1]) != 1:
                    p_hat = float(el[1]) / sample_size
                    p_zero = (num_of_resampling + 1.0) / virtual_dataset_size
                    z_alpha = stats.norm.ppf(confidence_value)
                    z_score = abs(p_hat - p_zero) / math.sqrt(p_zero * (1 - p_zero) / sample_size)
                    if ind == -1:  # Add new probability to the distribution
                        if z_score > z_alpha:
                            new_acception_prob.append((el[0], x))
                    else:  # Updating the probability
                        if z_score > z_alpha:
                            new_acception_prob[ind] = (el[0], new_acception_prob[ind][1] * x)
                    virtual_dataset_size = virtual_dataset_size - (1.0 / float(x)) + 1
        new_virtual_dataset_size = virtual_dataset_size

        return new_acception_prob, new_virtual_dataset_size

