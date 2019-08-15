import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

class ReportGenerator:
    def __init__(self, report_root, model_name):
        if not os.path.exists(report_root):
            os.mkdir(report_root)
        self.report_path = report_root + "/" + model_name
        if os.path.exists(self.report_path):
            shutil.rmtree(self.report_path)
        os.mkdir(self.report_path)
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
            hit_spectrum = hits[name]['spectrum']
            compound = sample[peak_num  - 1]
            sample_spectrum = compound['spectrum']
            fig = plt.figure()
            plt.plot(sample_spectrum)
            plt.plot(-hit_spectrum)
            plt.xlabel('m/z')
            plt.ylabel('<---library/sample--->')
            conf = "Low"
            if (pred[i] == 1):
                conf = "Medium"
            elif (pred[i] == 2):
                conf = "High"
            title = str(peak_num) + ":" + compound['name'] + ", Confidence: " + conf
            plt.title(title)
            pdf.savefig(fig)
            plt.close()
            i = i + 1

    def report_pdf1(self, sample_name, hits, sample, pred):
        file_path = os.path.join(self.report_path, sample_name)
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        fig_path = os.path.join(file_path, "report.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)
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
            title = str(peak_num) + ":" + compound['name'] + ", Confidence: " + conf
            plt.title(title)
            pdf.savefig(fig)
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
