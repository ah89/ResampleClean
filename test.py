import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from duplication_data_creater import DataCreater
from resample_clean import ResampleClean, ResampleCleanWithHypothesis
from sample_distribution_learn import SamplingDistributionFinder


class SampleCleanTest:

    def __init__(self, number_of_experiments, number_of_repeat, min_range, max_range):
        self.number_of_experiments = number_of_experiments
        self.number_of_repeat = number_of_repeat
        self.min_range = min_range
        self.max_range = max_range

    def data_generator(self, duplication_rate, data_size):
        return DataCreater.create_data(data_size, 1-duplication_rate, self.min_range, self.max_range)

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
                    our_data = self.data_generator(duplication_rate, data_size)
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

    def stat_test_precision(self, data_size, min_sam_size ,step_size, max_sam_size , list_of_dup):
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
                    our_data = self.data_generator(duplication_rate, data_size)
                    # our_data = [1,1,1,1,3,3,3,3,3,2,4]
                    y = np.mean(list(set(our_data)))
                    sc = ResampleCleanWithHypothesis(data=our_data, sample_size=sample_size, confidence_value=0.95)

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
        self.hist(result, x_point, list_of_dup)

    def hist(self, result, x_point, list_of_dup):
        pdf = matplotlib.backends.backend_pdf.PdfPages("error.pdf")
        fig = plt.figure(111)
        plt.title("Estimator error with different duplication rate")
        plt.xlabel("Relative sample size")
        plt.ylabel("Estimator's error")
        for enum in range(len(list_of_dup)):
            plt.plot(x_point, result[enum], label='r='+str(list_of_dup[enum]))
        leg = plt.legend(loc='best', ncol=len(list_of_dup), mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.3)
        plt.grid(True)
        pdf.savefig(fig)
        pdf.close()
        plt.show()

    def dup_resamp(self, data_size, min_sam_size, step, max_sam_size, list_of_dup):
        result = []
        sample_size = min_sam_size
        sam_size_point = []
        print "Start Test :"
        while sample_size < max_sam_size:
            print "Start test for sample size:"+str(sample_size)
            for_this_dup = []
            for duplication_rate in list_of_dup:
                print "duplication rate:" + str(duplication_rate)
                re_mean = []
                for i in range(self.number_of_experiments):
                    our_data = self.data_generator(duplication_rate, data_size)
                    # our_data = [1,1,1,1,3,3,3,3,3,2,4]
                    y = np.mean(list(set(our_data)))
                    sc = ResampleClean(our_data, sample_size)
                    re = sc.truth_sample()
                    re_mean.append(re)
                for_this_dup.append(np.mean(re_mean))
                print "Value :"+str(np.mean(re_mean))
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
        self.hist_dup_vs_number(result, sam_size_point, list_of_dup)

    def dup_resamp_stat(self, data_size, min_sam_size, step, max_sam_size, list_of_dup):
        result = []
        sample_size = min_sam_size
        sam_size_point = []
        print "Start Test :"
        while sample_size < max_sam_size:
            print "Start test for sample size:"+str(sample_size)
            for_this_dup = []
            for duplication_rate in list_of_dup:
                print "duplication rate:" + str(duplication_rate)
                re_mean = []
                for i in range(self.number_of_experiments):
                    our_data = self.data_generator(duplication_rate, data_size)
                    # our_data = [1,1,1,1,3,3,3,3,3,2,4]
                    y = np.mean(list(set(our_data)))
                    sc = ResampleCleanWithHypothesis(data=our_data, sample_size=sample_size, confidence_value=0.95)
                    re = sc.learning_termination()
                    re_mean.append(re)
                for_this_dup.append(np.mean(re_mean))
                print "Value :"+str(np.mean(re_mean))
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
        self.hist_dup_vs_number(result, sam_size_point, list_of_dup)

    def hist_dup_vs_number(self, result, sam_size_point, list_of_dup):
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

    def dupropy_error(self, data_size, resampling_iteration, sample_size, list_of_dup):
        print "Test Correlation dupropy and error started :\n"
        result = []
        for duplication_rate in list_of_dup:
            print "For duplication rate : "+str(duplication_rate)
            our_data = self.data_generator(duplication_rate, data_size)
            rc = ResampleClean(our_data, sample_size)
            y = np.mean(list(set(our_data)))
            err = []
            dupropy = []
            for re in range(resampling_iteration):
                # threshold_tau = rc.dupropy_gain(SamplingDistributionFinder.sampling(rc.data, 2 * rc.sample_size),
                #                             rc.acception_dist)
                rc.resampling_novelty(0.25)
                threshold_tau = rc.dupropy_gain(rc.data, rc.acception_dist)
                dupropy.append(threshold_tau)
                trys = []
                for rep_count in range(self.number_of_repeat):
                    t = rc.distributed_sampler(rc.acception_dist)
                    # trys.append(abs(np.mean(t)-y))
                    # trys.append(abs(np.mean(list(set(t)))-y)/y)
                    trys.append(np.mean(t))
                x = np.mean(trys)
                err.append(abs(x-y)/y)
            fx = open('dupopyg_for_dup' + str(int(duplication_rate*100)) + '.csv', 'w')
            for ele in dupropy:
                fx.write(str(ele) + '\n')
            fx.close()
            f = open('err_for_dup' + str(int(duplication_rate*100)) + '.csv', 'w')
            for ele in err:
                f.write(str(ele) + '\n')
            f.close()
            result.append([dupropy,err])

        return result

    def improve(self, data_size, resampling_iteration, sample_size, list_of_dup):
        print "Test Correlation dupropy and error started :\n"
        result = []
        for duplication_rate in list_of_dup:
            print "For duplication rate : "+str(duplication_rate)
            our_data = self.data_generator(duplication_rate, data_size)
            rc = ResampleClean(our_data, sample_size)
            y = np.mean(list(set(our_data)))
            err = []
            dupropy = []
            for re in range(resampling_iteration):
                # threshold_tau = rc.dupropy_gain(SamplingDistributionFinder.sampling(rc.data, 2 * rc.sample_size),
                #                             rc.acception_dist)
                rc.resampling()
                trys = []
                for rep_count in range(self.number_of_repeat):
                    t = rc.distributed_sampler(rc.acception_dist)
                    trys.append(np.mean(t))
                x = np.mean(trys)
                err.append(abs(x-y)/y)
            fx = open('dupopyg_for_dup' + str(int(duplication_rate*100)) + '.csv', 'w')
            for ele in dupropy:
                fx.write(str(ele) + '\n')
            fx.close()
            f = open('err_for_dup' + str(int(duplication_rate*100)) + '.csv', 'w')
            for ele in err:
                f.write(str(ele) + '\n')
            f.close()
            result.append([dupropy,err])

        return result


test = SampleCleanTest(number_of_experiments=500, number_of_repeat=500 , min_range=500, max_range=10000)



# list3 = [0.1, 0.2, 0.3, 0.4, 0.5]
# list_of_dup = [0.1, 0.3]
list2 = [0.1, 0.15, 0.2, 0.25, 0.3]
listac = [0.2, 0.3]
# # test.general_test(1000, 0.9, 50)
# test.precision_test(10000, 50, 200, 1000, list2)
test.dup_resamp(10000, 200, 200, 1001, list2)
# test.stat_test_precision(10000, 50, 200, 1000, list2)
# test.dupropy_error(data_size=1000,resampling_iteration=100,sample_size=100,list_of_dup=list2)