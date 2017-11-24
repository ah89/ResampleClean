import duplication_data_creater as dcreate
from sample_distribution_learn import SamplingDistributionFinder
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf


class ResampleClean:
    def __init__(self, data, sample_size):

        self.sample_size = sample_size
        self.data = data
        self.sample = SamplingDistributionFinder.sampling(data, self.sample_size)
        # print self.sample
        sample_dist = SamplingDistributionFinder.sample_distribution(self.sample)
        # print sample_dist
        self.acception_dist = SamplingDistributionFinder.resample_value_acception_distribution(data, sample_dist, None)
        # print self.acception_dist

    def resampling(self):

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
            tmpresam = SamplingDistributionFinder.sampling(self.data, tmp_resample_size)
        self.sample = resample
        # print resample
        sample_dist = SamplingDistributionFinder.sample_distribution(self.sample)
        self.acception_dist = SamplingDistributionFinder.resample_value_acception_distribution(self.data, sample_dist,
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
        while self.is_repeated(self.sample):
            self.resampling()
            stop += 1
        return stop


class SampleCleanTest:

    def __init__(self, number_of_experiments, number_of_repeat):
        self.number_of_experiments = number_of_experiments
        self.number_of_repeat = number_of_repeat

    def data_generator(self, duplication_rate, data_size):
        return dcreate.DataCreater(data_size, duplication_rate).data_list()

    def general_test(self, data_size, duplication_rate, sample_size):
        re_mean = []
        sd = []
        imp = []
        for i in range(self.number_of_experiments):
            print "\n Experiment number "+str(i+1)
            our_data = self.data_generator(duplication_rate, data_size)
            # our_data = [1,1,1,1,3,3,3,3,3,2,4]
            print "The real average is: "
            y = np.mean(list(set(our_data)))
            print y
            sc = ResampleClean(our_data, sample_size)
            print "Without correction error is:"
            tmp_sum = 0
            for rep_count in range(self.number_of_repeat):
                tmp_sum += abs(np.mean(SamplingDistributionFinder.sampling(our_data, sample_size))-y)/y
            print tmp_sum/self.number_of_repeat

            re = sc.truth_sample()
            trys = []
            for rep_count in range(self.number_of_repeat):
                trys.append(sc.distributed_sampler(sc.acception_dist))
            print "Sample result : "
            x = np.mean(trys)
            print x

            print "Error rate is :"
            print abs(x-y)/y

            tmp_imp = (tmp_sum / self.number_of_repeat) - (abs(x - y) / y)
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

    def precision_test(self, data_size, min_sam_size ,step_size, max_sam_size , list_of_dup):
        pdf = matplotlib.backends.backend_pdf.PdfPages("error.pdf")
        result = []
        x_point = []
        for duplication_rate in list_of_dup:
            sample_size = min_sam_size
            for_this_dup_rate = []
            sam_size_point = []
            while sample_size < max_sam_size:
                sd = []
                for i in range(self.number_of_experiments):
                    our_data = self.data_generator(duplication_rate, data_size)
                    # our_data = [1,1,1,1,3,3,3,3,3,2,4]
                    y = np.mean(list(set(our_data)))
                    sc = ResampleClean(our_data, sample_size)

                    trys = []
                    for rep_count in range(self.number_of_repeat):
                        trys.append(sc.distributed_sampler(sc.acception_dist))
                    x = np.mean(trys)

                    sd.append(abs(x - y) / y)

                for_this_dup_rate.append(np.mean(sd))
                sam_size_point.append(float(sample_size)/data_size)
                print "Sample size is :"+str(sample_size)
                sample_size += step_size
            x_point = sam_size_point
            result.append(for_this_dup_rate)
            print "Rate for "+str(duplication_rate)+" has been done!"
        fig = plt.figure(111)
        plt.title("Estimator error with different duplication rate")
        plt.xlabel("Relative sample size")
        plt.ylabel("Estimator's error")
        for enum in range(len(list_of_dup)):
            plt.plot(x_point,result[enum], label='r='+str(list_of_dup[enum]))
        leg = plt.legend(loc='best', ncol=len(list_of_dup), mode="expand", shadow=True, fancybox=True)
        # leg = plt.legend(loc='upper right', ncol=len(list_of_dup), mode="None", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.3)
        # plt.show()
        plt.grid(True)
        pdf.savefig(fig)
        pdf.close()


test = SampleCleanTest(5000, 150)
list3 = [0.1, 0.2, 0.3, 0.4, 0.5]
list_of_dup = [0.1, 0.3, 0.5, 0.7]
list2 = [0.1, 0.3]
test.precision_test(10000, 50,50, 5000, list3)
