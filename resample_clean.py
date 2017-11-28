from duplication_data_creater import DataCreater
from sample_distribution_learn import SamplingDistributionFinder
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import collections
import math
from scipy.stats.stats import pearsonr
import copy


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
        prev_dist = []
        while len(list(set(self.acception_dist)-set(prev_dist))) != 0:
            prev_dist = copy.copy(self.acception_dist)
            self.resampling()
            stop += 1
        return stop

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




class SampleCleanTest:

    def __init__(self, number_of_experiments, number_of_repeat, min_range, max_range):
        self.number_of_experiments = number_of_experiments
        self.number_of_repeat = number_of_repeat
        self.min_range = min_range
        self.max_range = max_range

    def data_generator(self, duplication_rate, data_size):
        return DataCreater.create_data(data_size, duplication_rate, self.min_range, self.max_range)
    def general_test(self, data_size, duplication_rate, sample_size):
        re_mean = []
        sd = []
        imp = []
        for i in range(self.number_of_experiments):
            print "\n Experiment number "+str(i+1)
            our_data = self.data_generator(1 - duplication_rate, data_size)
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
                trys.append(np.mean(list(set(sc.distributed_sampler(sc.acception_dist)))))
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
        result = []
        x_point = []
        print "TEST STARTED"
        for duplication_rate in list_of_dup:
            print "Test for "+str(duplication_rate)+" of duplications"
            sample_size = min_sam_size
            for_this_dup_rate = []
            sam_size_point = []
            while sample_size < max_sam_size:
                sd = []
                for i in range(self.number_of_experiments):
                    our_data = self.data_generator(1 - duplication_rate, data_size)
                    # our_data = [1,1,1,1,3,3,3,3,3,2,4]
                    y = np.mean(list(set(our_data)))
                    sc = ResampleClean(our_data, sample_size)

                    trys = []
                    for rep_count in range(self.number_of_repeat):
                        t = sc.distributed_sampler(sc.acception_dist)
                        trys.append(np.mean(list(set(t))))
                    x = np.mean(trys)

                    sd.append(abs(x - y) / y)

                for_this_dup_rate.append(np.mean(sd))
                sam_size_point.append(float(sample_size)/data_size)
                print "Sample size is :"+str(sample_size)
                sample_size += step_size
            x_point = sam_size_point
            f = open('prec_values'+str(int(duplication_rate*100))+'.txt', 'w')
            for ele in for_this_dup_rate:
                f.write(str(ele) + '\n')
            f.close()
            result.append(for_this_dup_rate)
            print "Rate for "+str(duplication_rate)+" has been done!"
        fx = open('prec_x_point.txt', 'w')
        for ele in x_point:
            fx.write(str(ele) + '\n')
        fx.close()
        self.hist(result,x_point)

    def hist(self, result, x_point, list_of_dup):
        pdf = matplotlib.backends.backend_pdf.PdfPages("error.pdf")
        fig = plt.figure(111)
        plt.title("Estimator error with different duplication rate")
        plt.xlabel("Relative sample size")
        plt.ylabel("Estimator's error")
        for enum in range(len(list_of_dup)):
            plt.plot(x_point,result[enum], label='r='+str(list_of_dup[enum]))
        leg = plt.legend(loc='best', ncol=len(list_of_dup), mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.3)
        plt.grid(True)
        pdf.savefig(fig)
        pdf.close()
        plt.show()

    def dup_resamp(self, data_size, min_sam_size, step, max_sam_size, list_of_dup):
        result = []
        sample_size = min_sam_size
        sam_size_point=[]
        print "Start Test :"
        while sample_size < max_sam_size:
            print "Start test for sample size:"+str(sample_size)
            for_this_dup = []
            for duplication_rate in list_of_dup:
                print "duplication rate:" + str(duplication_rate)
                re_mean = []
                for i in range(self.number_of_experiments):
                    our_data = self.data_generator(1-duplication_rate, data_size)
                    # our_data = [1,1,1,1,3,3,3,3,3,2,4]
                    y = np.mean(list(set(our_data)))
                    sc = ResampleClean(our_data, sample_size)
                    re = sc.truth_sample()
                    re_mean.append(re)
                for_this_dup.append(np.mean(re_mean))
            f = open('dupVSresam_for_samsize'+str(int(sample_size))+'.txt', 'w')
            for ele in for_this_dup:
                f.write(str(ele) + '\n')
            f.close()
            result.append(for_this_dup)
            sam_size_point.append(sample_size)
            sample_size += step
        fx = open('dup_x_point.txt', 'w')
        for ele in list_of_dup:
            fx.write(str(ele) + '\n')
        fx.close()
        self.hist_dup_vs_number(result,sam_size_point,list_of_dup)


    def hist_dup_vs_number(self,result,sam_size_point,list_of_dup):
        pdf = matplotlib.backends.backend_pdf.PdfPages("dup-vs-number-resampling.pdf")
        fig = plt.figure(111)
        plt.title("Number of resampling vs. duplication rate in different sample size")
        plt.xlabel("Duplication rate")
        plt.ylabel("Number of resampling")
        for enum in range(len(sam_size_point)):
            plt.plot(list_of_dup, result[enum], label='S=' + str(sam_size_point[enum]))
        leg = plt.legend(loc='best', ncol=len(sam_size_point), mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.3)
        plt.grid(True)
        pdf.savefig(fig)
        pdf.close()
        plt.show()








test = SampleCleanTest(500, 100 ,500, 10000)



# list3 = [0.1, 0.2, 0.3, 0.4, 0.5]
# list_of_dup = [0.1, 0.3]
list2 = [0.1, 0.15, 0.2, 0.25, 0.3]
listac = [0.2, 0.3]
# # test.general_test(1000, 0.9, 50)
# test.precision_test(10000, 50, 200, 1000, listac)
test.dup_resamp(10000, 200, 200, 1001, list2)
