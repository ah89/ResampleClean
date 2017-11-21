import random as rand
import math


class DataCreater:
    def __init__(self, number_of_data, rate_of_duplication):
        self.number_of_data = number_of_data
        self.rate_of_duplication = rate_of_duplication

    def _find_range(self):
        """
        This function create range from 1 to maximum that by considering the rate
        of duplications we can have maximum number that we can see
        :return: int
        """

        max_of_range = int(math.floor(self.number_of_data / (1 + self.rate_of_duplication)))
        return max_of_range

    def data_list(self):
        """
        This method by using the maximum range return list with rate_of_duplication number of duplication
        :return:
        """
        max_range = self._find_range()

        data = []
        for i in range(1, self.number_of_data):
            data.append(rand.randint(1, max_range))
        return data

    def frequencies(self, data_list):
        freq = [0] * self._find_range()

        for element in data_list:
            freq[element - 1] += 1
        return freq
