import duplication_data_creater as dcreate
from sample_distribution_learn import SamplingDistributionFinder
import random
import numpy as np





class ResampleClean:
    def __init__(self, data, sample_size):

        self.sample_size = sample_size
        self.data = data
        self.sample = SamplingDistributionFinder.sampling(data, sample_size)
        # print self.sample
        sample_dist = SamplingDistributionFinder.sample_distribution(self.sample)
        # print sample_dist
        self.acception_dist = SamplingDistributionFinder.resample_value_acception_distribution(data, sample_dist, None)
        # print self.acception_dist

    def resampling(self):

        tmpresam = SamplingDistributionFinder.sampling(self.data, self.sample_size)

        resample = []
        while (len(resample) != self.sample_size):
            inclusion_map = [0] * len(tmpresam)
            for el in tmpresam:
                rand = random.random()
                acception=self._find_acc_prob(el, self.acception_dist)
                if rand < acception:
                    inclusion_map[tmpresam.index(el)] = 1
            for i in range(len(tmpresam)):
                if inclusion_map[i] == 1:
                    resample.append(tmpresam[i])
            tmp_resample_size = self.sample_size - len(resample)
            tmpresam = SamplingDistributionFinder.sampling(self.data,tmp_resample_size)
        self.sample = resample
        # print resample
        sample_dist = SamplingDistributionFinder.sample_distribution(self.sample)
        self.acception_dist = SamplingDistributionFinder.resample_value_acception_distribution(self.data,sample_dist,
                                                                                               self.acception_dist)

    def distributed_sampler(self, accept_prob):
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
            tmpresam = SamplingDistributionFinder.sampling(self.data, tmp_resample_size)
        return resample

    def _find_acc_prob(self,value,acception_dist):
        for elemnt in acception_dist:
            if elemnt[0] == value:
                return elemnt[1]
        return 1

    def is_repeated(self,sample):
        if len(list(set(sample))) == len(sample):
            return 0
        return 1


    def truth_sample(self):
        stop=0
        while self.is_repeated(self.sample) :
            self.resampling()
            stop+=1
        return stop
re_mean=[]
sd=[]
imp=[]
for i in range(30):
    sample_size = 50
    number_of_repeat = 1000
    print "\n Experiment number "+str(i+1)
    our_data = dcreate.DataCreater(1000, 0.9).data_list()
    # our_data = [1,1,1,1,3,3,3,3,3,2,4]
    print "The real average is: "
    y = np.mean(list(set(our_data)))
    print y
    sc = ResampleClean(our_data, sample_size)
    print "Without correction error is:"
    tmp_sum = 0
    for i in range(number_of_repeat):
        tmp_sum += abs(np.mean(SamplingDistributionFinder.sampling(our_data, sample_size))-y)/y
    print tmp_sum/number_of_repeat

    re = sc.truth_sample()
    trys = []
    for i in range(number_of_repeat):
        trys.append(sc.distributed_sampler(sc.acception_dist))
    print "Sample result : "
    x = np.mean(trys)
    print x

    print "Error rate is :"
    print abs(x-y)/y

    tmp_imp = (tmp_sum / number_of_repeat) - (abs(x - y) / y)
    print "The improvment is: "
    print tmp_imp


    print "Number of resampling: "
    print re


    imp.append(tmp_imp)
    re_mean.append(re)
    sd.append(abs(x - y) / y)

print "\n Summary :"

print "Average of resampling is : "
print np.mean(re_mean)

print "Average error of etimator:"
print np.mean(sd)

print "Average improvment of etimator:"
print np.mean(imp)
