import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import sys

class ReportGenerator:
    def __init__(self, report_root, model_name):
        if not os.path.exists(report_root):
            os.mkdir(report_root)
        self.report_path = report_root + "/" + model_name
        if os.path.exists(self.report_path):
            shutil.rmtree(self.report_path)
        os.mkdir(self.report_path)
        self.image_path = self.report_path + '/' + 'plots' #Added
        if os.path.exists(self.image_path):        #Added
            shutil.rmtree(self.image_path)          #Added
        os.mkdir(self.image_path)                  #Added
    def report_csv(self, sample_name, test, pred):
        file_path = os.path.join(self.report_path, sample_name)
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        rep_file_path = os.path.join(file_path, "report.tsv");
        f = open(rep_file_path, "a+")
        f.write("#peak_num\tname\tconfidence\n")
        i = 0
        for (peak_num, name, confidence) in test:
            f.write("{peak_num}\t{name}\t{pred}\n".format(peak_num=peak_num, name=name,
                                                                 pred=pred[i]))
            i = i + 1
        print("Reports are generated under {file_path}".format(file_path=file_path))
        f.close()

    def report_pdf(self, sample_name, hits, sample, valid_gt, pred):
        file_path = os.path.join(self.report_path, sample_name)
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        fig_path = os.path.join(file_path, "report.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)
        i = 0;
        for (peak_num, name, confidence) in valid_gt:
            compound = sample[peak_num - 1]
            hit_last_index = np.nonzero(hits[name]['spectrum'])[0][-1]
            sample_last_index = np.nonzero(compound['spectrum'])[0][-1]
            max_index = max(hit_last_index, sample_last_index)
            hit_spectrum = hits[name]['spectrum'][0:max_index]
            sample_spectrum = compound['spectrum'][0:max_index]
            fig = plt.figure()
            t1,stemlines1,_t2 = plt.stem(sample_spectrum, markerfmt=" ")
            plt.setp(stemlines1, linewidth=0.5, color='cornflowerblue')
            t1, stemlines, _t2 = plt.stem(-hit_spectrum, markerfmt=" ")
            plt.setp(stemlines, linewidth=0.5, color='coral')
            plt.xlabel('m/z')
            plt.ylabel('<---library/sample--->')
            conf = "Low"
            if (pred[i] == 1):
                conf = "Medium"
            elif (pred[i] == 2):
                conf = "High"
            title = str(peak_num) + ":" + compound['name'] + ",\n Confidence: " + conf
            file_name = str(peak_num) + '.png'  # Added
            plt.title(title)
            #pdf.savefig(fig) Removed
            plt.savefig(os.path.join(self.image_path, file_name)) #Added
            plt.close()
            i = i + 1
            percent = i * 100/len(valid_gt)
            print("\rReport progress: {:0.2f} %".format(percent), end='')
            sys.stdout.flush()


    def report_pdf1(self, sample_name, hits, sample, pred):
        file_path = os.path.join(self.report_path, sample_name)
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        fig_path = os.path.join(file_path, "report.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)
        i = 0
        for (peak_num, name, confidence) in pred:
            hit_spectrum = hits[name]['spectrum']
            compound = sample[peak_num - 1]
            sample_spectrum = compound['spectrum']
            fig = plt.figure()
            plt.plot(sample_spectrum)
            plt.plot(-hit_spectrum)
            plt.xlabel('m/z')
            plt.ylabel('<---library/sample--->')
            conf = "Low"
            if (confidence == 1):
                conf = "Medium"
            elif (confidence == 2):
                conf = "High"
            title = str(peak_num) + ":" + compound['name'] + ",\n Confidence: " + conf
            file_name = str(peak_num) + '.png' #Added
            plt.title(title)
            #pdf.savefig(fig) Removed
            plt.savefig(os.path.join(self.image_path, file_name)) #Added
            plt.close()


    def report_csv1(self, sample_name, pred):
        file_path = os.path.join(self.report_path, sample_name)
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        rep_file_path = os.path.join(file_path, "report.tsv");
        f = open(rep_file_path, "a+")
        f.write("#peak_num\tname\tconfidence\n")
        for (peak_num, name, confidence) in pred:
            f.write("{peak_num}\t{name}\t{pred}\n".format(peak_num=peak_num, name=name,
                                                       pred=confidence))
        print("Reports are generated under {file_path}".format(file_path=file_path))
        f.close()

    def __del__(self):
        pass
