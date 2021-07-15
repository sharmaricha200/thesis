import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import glob


def scatter(path_to_CINeMA, path_to_csvs, all_peak_nums):
    fig_path = os.path.join(os.path.dirname(path_to_CINeMA), 'Molecular_Weight_vs_Retention_Times.pdf')
    csvs_path = '{}/*.csv'.format(path_to_csvs)
    pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)

    z = 0

    fig = plt.figure()

    for file, peak_list in zip(glob.glob(csvs_path), all_peak_nums):
        df = pd.read_csv(file, encoding = 'utf-8')
        x_axis = df['Molecular Weight']
        y_axis1 = df['1st RT']
        peak_num = df['Peak Number']

        for peak, x, y1 in zip(peak_num, x_axis, y_axis1):
            if peak in peak_list:
                z = z + 1
                plt.scatter(x, y1, color = 'blue')
        plt.xlabel('Molecular Weight (g/mol)')
        plt.ylabel('1st Retention Time (s)')

    print(z)
    pdf.savefig(fig)

    fig2 = plt.figure()

    z = 0

    for file, peak_list in zip(glob.glob(csvs_path), all_peak_nums):
        df = pd.read_csv(file, encoding='utf-8')
        x_axis = df['Molecular Weight']
        y_axis2 = df['2nd RT']
        peak_num = df['Peak Number']

        for peak, x, y2 in zip(peak_num, x_axis, y_axis2):
            if peak in peak_list:
                z = z + 1
                plt.scatter(x, y2, color = 'blue')
        plt.xlabel('Molecular Weight (g/mol)')
        plt.ylabel('2nd Retention Time (s)')

    print(z)

    pdf.savefig(fig2)

    pdf.close()
