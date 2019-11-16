import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf


def scatter(path_to_CINeMA, path_to_csv):
    fig_path = os.path.join(os.path.dirname(path_to_CINeMA), 'Molecular_Weight_vs_Retention_Times.pdf')
    pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)

    df = pd.read_csv(path_to_csv, encoding = 'latin-1')
    x_axis = df['Molecular Weight']
    y_axis1 = df['1st RT']
    y_axis2 = df['2nd RT']

    fig = plt.figure()
    plt.scatter(x_axis, y_axis1)
    plt.xlabel('Molecular Weight (g/mol)')
    plt.ylabel('1st Retention Time (s)')
    pdf.savefig(fig)

    fig = plt.figure()
    plt.scatter(x_axis, y_axis2)
    plt.xlabel('Molecular Weight (g/mol)')
    plt.ylabel('2nd Retention Time (s)')
    pdf.savefig(fig)

    pdf.close()