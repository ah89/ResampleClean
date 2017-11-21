import random


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
    def resample_value_acception_distribution(data, distribution, prev_acception_prob = None):
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
        sample_size = sum(freq)
        data_size = len(data)
        if prev_acception_prob == None:
            new_acception_prob = []
            for el in distribution:
                # x = float(sample_size) / float(el[1] * data_size)
                x = 1 / float(el[1])
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
                    new_acception_prob.append((el[0], x))
                else:
                    # x = float(sample_size) / float(el[1] * data_size)
                    x = 1 / float(el[1])
                    new_acception_prob[ind] = (el[0], new_acception_prob[ind][1] * x)
        return new_acception_prob
