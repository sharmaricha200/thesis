import os
import seaborn as sns
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
    def report_csv(self, test, pred):
        file_path = os.path.join(self.report_path, 'all_data')
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        rep_file_path = os.path.join(file_path, "report.tsv");
        f = open(rep_file_path, "a+")
        f.write("#peak_num\tname\tconfidence\n")
        i = 0
        for j in test:
            for (peak_num, name, confidence, molecular_ion) in j:
                f.write("{peak_num}\t{name}\t{pred}\n".format(peak_num=peak_num, name=name,
                                                                     pred=pred[i]))
                i = i + 1
        print("Reports are generated under {file_path}".format(file_path=file_path))
        f.close()

    def report_pdf(self, hits, sample, valid_gt, pred, names):
        file_path = os.path.join(self.report_path, 'all_data')
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        fig_path_correct = os.path.join(file_path, "report_correct.pdf")
        fig_path_incorrect = os.path.join(file_path, "report_incorrect.pdf")
        pdf_correct = matplotlib.backends.backend_pdf.PdfPages(fig_path_correct)
        pdf_incorrect = matplotlib.backends.backend_pdf.PdfPages(fig_path_incorrect)
        i = 0
        high_correct = 0
        high_incorrect = 0
        low_correct = 0
        low_incorrect = 0
        for v, s, h, n in zip(valid_gt, sample, hits, names):
            for (peak_num, name, confidence, molecular_ion) in v:
                compound = s[peak_num - 1]
                hit_last_index = np.nonzero(h[name]['spectrum'])[0][-1]
                sample_last_index = np.nonzero(compound['spectrum'])[0][-1]
                max_index = max(hit_last_index, sample_last_index)
                hit_spectrum = h[name]['spectrum'][0:max_index + 1]
                hit_spectrum[0:49] = 0
                sample_spectrum = compound['spectrum'][0:max_index + 1]
                fig = plt.figure()
                x = [i for i in range(1, len(sample_spectrum) + 1)]
                t1,stemlines1,_t2 = plt.stem(x, sample_spectrum, markerfmt=" ", use_line_collection=True)
                plt.setp(stemlines1, linewidth=0.5, color='cornflowerblue')
                x = [i for i in range(1, len(hit_spectrum) + 1)]
                t1, stemlines, _t2 = plt.stem(x, -hit_spectrum, markerfmt=" ", use_line_collection=True)
                plt.setp(stemlines, linewidth=0.5, color='coral')
                plt.xlabel('m/z')
                plt.ylabel('<---library/sample--->')
                conf = "Low"
                if (pred[i] == 1):
                    conf = "Medium"
                elif (pred[i] == 2):
                    conf = "High"
                title = n + ':' + str(peak_num) + ":" + compound['name'] + ",\n Confidence: " + conf
                plt.title(title)
                if pred[i] == confidence:
                    pdf_correct.savefig(fig)
                    if conf == 'High':
                        high_correct = high_correct + 1
                    if conf == 'Low':
                        low_correct = low_correct + 1
                elif pred[i] != confidence:
                    pdf_incorrect.savefig(fig)
                    if conf == 'High':
                        high_incorrect = high_incorrect + 1
                    elif conf == 'Low':
                        low_incorrect = low_incorrect + 1
                plt.close()
                i = i + 1
                percent = i * 100/len(pred)
                print("\rReport progress: {:0.2f} %".format(percent), end='')
                sys.stdout.flush()
        high_correct_perc = high_correct / (high_correct + low_correct)
        low_correct_perc = low_correct / (high_correct + low_correct)
        high_incorrect_perc = high_incorrect / (high_incorrect + low_incorrect)
        low_incorrect_perc = low_incorrect / (high_incorrect + low_incorrect)
        print('\n')
        print(high_correct_perc, high_correct, low_correct_perc, low_correct, high_incorrect_perc, high_incorrect, low_incorrect_perc, low_incorrect)
        pdf_correct.close()
        pdf_incorrect.close()

    def report_csv3(self, test, pred, indicies):
        file_path = os.path.join(self.report_path, 'all_data')
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        rep_file_path = os.path.join(file_path, "report.tsv");
        f = open(rep_file_path, "a+")
        f.write("#peak_num\tname\tconfidence\n")
        i = 0
        my_dict = {}
        for compound, rank in zip(indicies, pred):
            my_dict[compound] = rank
        for j in test:
            for (peak_num, name, confidence, molecular_ion) in j:
                if i in indicies:
                    f.write("{peak_num}\t{name}\t{pred}\n".format(peak_num=peak_num, name=name, pred=my_dict[i]))
                    i = i + 1
                else:
                    i = i + 1
        print("Reports are generated under {file_path}".format(file_path=file_path))
        f.close()

    def report_pdf3(self, hits, sample, valid_gt, pred, indicies):
        file_path = os.path.join(self.report_path, 'all_data')
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        fig_path = os.path.join(file_path, "report.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)
        i = 0
        my_dict = {}
        for compound, rank in zip(indicies, pred):
            my_dict[compound] = rank
        for v, s, h in zip(valid_gt, sample, hits):
            for (peak_num, name, confidence, molecular_ion) in v:
                if i in indicies:
                    compound = s[peak_num - 1]
                    hit_last_index = np.nonzero(h[name]['spectrum'])[0][-1]
                    sample_last_index = np.nonzero(compound['spectrum'])[0][-1]
                    max_index = max(hit_last_index, sample_last_index)
                    hit_spectrum = h[name]['spectrum'][0:max_index + 1]
                    hit_spectrum[0:49] = 0
                    sample_spectrum = compound['spectrum'][0:max_index + 1]
                    fig = plt.figure()
                    x = [i for i in range(1, len(sample_spectrum) + 1)]
                    t1, stemlines1, _t2 = plt.stem(x, sample_spectrum, markerfmt=" ", use_line_collection=True)
                    plt.setp(stemlines1, linewidth=0.5, color='cornflowerblue')
                    x = [i for i in range(1, len(hit_spectrum) + 1)]
                    t1, stemlines, _t2 = plt.stem(x, -hit_spectrum, markerfmt=" ", use_line_collection=True)
                    plt.setp(stemlines, linewidth=0.5, color='coral')
                    plt.xlabel('m/z')
                    plt.ylabel('<---library/sample--->')
                    conf = "Low"
                    if (my_dict[i] == 1):
                        conf = "Medium"
                    elif (my_dict[i] == 2):
                        conf = "High"
                    title = str(peak_num) + ":" + compound['name'] + ",\n Confidence: " + conf
                    plt.title(title)
                    pdf.savefig(fig)
                    plt.close()
                    i = i + 1
                    # percent = i * 100 / len(pred)
                    # print("\rReport progress: {:0.2f} %".format(percent), end='')
                    sys.stdout.flush()
                else:
                    i = i + 1
        pdf.close()

    def report_pdf1(self, hits, sample, compounds, pred, names):
        file_path = os.path.join(self.report_path, 'all_data')
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        fig_path = os.path.join(file_path, "report.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)
        i = 0
        for v, s, h, n in zip(compounds, sample, hits, names):
            for (peak_num, name, molecular_ion) in v:
                compound = s[peak_num - 1]
                hit_last_index = np.nonzero(h[name]['spectrum'])[0][-1]
                sample_last_index = np.nonzero(compound['spectrum'])[0][-1]
                max_index = max(hit_last_index, sample_last_index)
                hit_spectrum = h[name]['spectrum'][0:max_index + 1]
                hit_spectrum[0:49] = 0
                sample_spectrum = compound['spectrum'][0:max_index + 1]
                fig = plt.figure()
                x = [i for i in range(1, len(sample_spectrum) + 1)]
                t1,stemlines1,_t2 = plt.stem(x, sample_spectrum, markerfmt=" ", use_line_collection=True)
                plt.setp(stemlines1, linewidth=0.5, color='cornflowerblue')
                x = [i for i in range(1, len(hit_spectrum) + 1)]
                t1, stemlines, _t2 = plt.stem(x, -hit_spectrum, markerfmt=" ", use_line_collection=True)
                plt.setp(stemlines, linewidth=0.5, color='coral')
                plt.xlabel('m/z')
                plt.ylabel('<---library/sample--->')
                conf = "Low"
                if (pred[i] == 1):
                    conf = "Medium"
                elif (pred[i] == 2):
                    conf = "High"
                title = n + ':' + str(peak_num) + ":" + compound['name'] + ",\n Confidence: " + conf
                plt.title(title)
                pdf.savefig(fig)
                plt.close()
                i = i + 1
                percent = i * 100/len(pred)
                print("\rReport progress: {:0.2f} %".format(percent), end='')
                sys.stdout.flush()
        pdf.close()

    def report_pdf2(self, sample_name, hits, sample, valid_gt, pred):
        file_path = os.path.join(self.report_path, sample_name)
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        fig_path = os.path.join(file_path, "report.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)
        i = 0
        for (peak_num, name, confidence, molecular_ion) in valid_gt:
            compound = sample[peak_num - 1]
            hit_last_index = np.nonzero(hits[name]['spectrum'])[0][-1]
            sample_last_index = np.nonzero(compound['spectrum'])[0][-1]
            max_index = max(hit_last_index, sample_last_index)
            hit_spectrum = hits[name]['spectrum'][0:max_index + 1]
            hit_spectrum[0:49] = 0
            sample_spectrum = compound['spectrum'][0:max_index + 1]
            fig = plt.figure()
            x = [i for i in range(1, len(sample_spectrum) + 1)]
            t1, stemlines1, _t2 = plt.stem(x, sample_spectrum, markerfmt=" ", use_line_collection=True)
            plt.setp(stemlines1, linewidth=0.5, color='cornflowerblue')
            x = [i for i in range(1, len(hit_spectrum) + 1)]
            t1, stemlines, _t2 = plt.stem(x, -hit_spectrum, markerfmt=" ", use_line_collection=True)
            plt.setp(stemlines, linewidth=0.5, color='coral')
            plt.xlabel('m/z')
            plt.ylabel('<---library/sample--->')
            conf = "Low"
            if (pred[i] == 1):
                conf = "Medium"
            elif (pred[i] == 2):
                conf = "High"
            title = str(peak_num) + ":" + compound['name'] + ",\n Confidence: " + conf
            plt.title(title)
            pdf.savefig(fig)
            plt.close()
            i = i + 1
            percent = i * 100 / len(valid_gt)
            print("\rReport progress: {:0.2f} %".format(percent), end='')
            sys.stdout.flush()
        pdf.close()

    def report_csv1(self, all_compounds, pred):
        file_path = os.path.join(self.report_path, 'all_data')
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        rep_file_path = os.path.join(file_path, "report.tsv");
        f = open(rep_file_path, "a+")
        f.write("#peak_num\tname\tconfidence\n")
        i = 0
        for j in all_compounds:
            for (peak_num, name, molecular_ion) in j:
                f.write("{peak_num}\t{name}\t{pred}\n".format(peak_num=peak_num, name=name,
                                                              pred=pred[i]))
                i = i + 1
        print("Reports are generated under {file_path}".format(file_path=file_path))
        f.close()

    def report_csv2(self, sample_name, test, pred):
        file_path = os.path.join(self.report_path, sample_name)
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        rep_file_path = os.path.join(file_path, "report.tsv");
        f = open(rep_file_path, "a+")
        f.write("#peak_num\tname\tconfidence\n")
        i = 0
        for (peak_num, name, confidence, molecular_ion) in test:
            f.write("{peak_num}\t{name}\t{pred}\n".format(peak_num=peak_num, name=name,
                                                          pred=pred[i]))
            i = i + 1
        print("Reports are generated under {file_path}".format(file_path=file_path))
        f.close()

    def report_matrix(self, sample_name, matrix):
        classes = ['Low', 'High']
        file_path = os.path.join(self.report_path, sample_name)
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        fig_path = os.path.join(file_path, 'matrix.pdf')
        pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)
        fig = plt.figure()
        sns.heatmap(matrix, annot=True, cbar=True, xticklabels=classes, yticklabels=classes, cmap='Blues', fmt = 'g')
        plt.ylabel('True Match Quality')
        plt.xlabel('Predicted Match Quality')
        pdf.savefig(fig)
        plt.close()
        pdf.close()

    def final_report_matrix(self, final_matrix):
        classes = ['Low', 'High']
        file_path = os.path.join(self.report_path, 'final_matrix')
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        fig_path = os.path.join(file_path, 'final_matrix.pdf')
        pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)
        fig = plt.figure()
        sns.heatmap(final_matrix, annot=True, cbar=True, xticklabels=classes, yticklabels=classes, cmap='Blues', fmt='g')
        plt.ylabel('True Match Quality')
        plt.xlabel('Predicted Match Quality')
        pdf.savefig(fig)
        plt.close()
        pdf.close()

    def __del__(self):
        pass
