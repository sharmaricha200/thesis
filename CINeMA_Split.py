"""
Usage:
  CINeMA_Split.py tts -d <data_dir> -s <model_save_file_path> [--w <data_dir>] [--e <epochs>]
  CINeMA_Split.py tts_rf_features -d <data_dir> -s <model_save_file_path> [--w <data_dir>]
  CINeMA_Split.py -h

Options:
  -h --help                     Show this screen.
  -d <data_dir>                 Data directory path.
  --w <data_dir>                Data directory path to be split
  --e <epochs>                  Number of epochs [default: 5]
  -s <model_save_file_path>     File path where model is saved
  --version                     Show version
"""

from docopt import docopt
import sys
import os

# Adding modules to python path, so that they can be imported.
ROOT_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.insert(0, ROOT_PATH + "/model/algorithmic")
sys.path.insert(0, ROOT_PATH + "/model/ml")
sys.path.insert(0, ROOT_PATH + "/utils")

import DataParser as dp
import ReportGenerator as rg
import numpy as np
import Features as ft
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def get_valid_gt(in_gt, sample):
    out_gt = []
    for (peak_num, name, confidence, molecular_ion) in in_gt:
        if name in hits and sample[peak_num - 1]['name'] == name:
            out_gt.append((peak_num, name, confidence, molecular_ion))
    return out_gt

def translate_pred(pred):
    ret  = []
    pred[pred <= 0.5] = 0
    pred[pred > 0.5] = 1

    for p in pred:
        if (p[0] == 0 and p[1] == 1):
            ret.append(2)
        elif (p[0] == 1 and p[1] == 0):
            ret.append(0)
        else:
            ret.append(-1)
    return ret

def get_matrix(pred, gt):
    TP = np.sum(np.logical_and(pred == 2, gt == 2))
    TN = np.sum(np.logical_and(pred == 0, gt == 0))
    FP = np.sum(np.logical_and(pred == 2, gt == 0))
    FN = np.sum(np.logical_and(pred == 0, gt == 2))
    confusion_matrix = np.matrix([[TN, FP], [FN, TP]])
    return confusion_matrix

if __name__ == '__main__':
    args = docopt(__doc__, version='1.0')
    if args['tts']:
        import MLModel as ml
        parser1 = dp.DataParser(args['-d'], dp.Mode.TRAIN)
        all_data = parser1.parseData()
        X = []
        Y = []
        all_names = []
        all_hits = []
        all_samples = []
        all_valid_gt = []
        for data in all_data:
            sample_name = data['name']
            hits = data['hits']
            sample = data['sample']
            ground_truth = data['ground_truth']
            valid_gt = get_valid_gt(ground_truth, sample)
            all_names.append(sample_name)
            all_hits.append(hits)
            all_samples.append(sample)
            all_valid_gt.append(valid_gt)
            for (peak_num, name, confidence, molecular_ion) in valid_gt:
                if confidence == 2:
                    gt = [0, 1]
                elif confidence == 0:
                    gt = [1, 0]
                sample_spectrum = sample[peak_num - 1]['spectrum']
                sample_spectrum = sample_spectrum / np.linalg.norm(sample_spectrum)
                hit_spectrum = hits[name]['spectrum']
                hit_spectrum = hit_spectrum / np.linalg.norm(hit_spectrum)
                hit_spectrum[0:49] = 0
                X.append(np.concatenate((sample_spectrum, hit_spectrum)))
                Y.append(gt)

        if args['--w']:
            parser2 = dp.DataParser(args['--w'], dp.Mode.TRAIN)
            all_data = parser2.parseData()
            X_2 = []
            Y_2 = []
            all_names = []
            all_hits = []
            all_samples = []
            all_valid_gt = []
            for data in all_data:
                sample_name = data['name']
                hits = data['hits']
                sample = data['sample']
                ground_truth = data['ground_truth']
                valid_gt = get_valid_gt(ground_truth, sample)
                all_names.append(sample_name)
                all_hits.append(hits)
                all_samples.append(sample)
                all_valid_gt.append(valid_gt)
                for (peak_num, name, confidence, molecular_ion) in valid_gt:
                    if confidence == 2:
                        gt = [0, 1]
                    elif confidence == 0:
                        gt = [1, 0]
                    sample_spectrum = sample[peak_num - 1]['spectrum']
                    sample_spectrum = sample_spectrum / np.linalg.norm(sample_spectrum)
                    hit_spectrum = hits[name]['spectrum']
                    hit_spectrum = hit_spectrum / np.linalg.norm(hit_spectrum)
                    hit_spectrum[0:49] = 0
                    X_2.append(np.concatenate((sample_spectrum, hit_spectrum)))
                    Y_2.append(gt)

            indicies = []
            for i in range(len(X_2)):
                indicies.append(i)

            X_train_2, X_test_2, y_train_2, y_test_2, indicies_train2, indicies_test2 = train_test_split(X_2, Y_2, indicies, test_size=0.20, random_state=100)

            for i in X_train_2:
                X.append(i)
            for i in y_train_2:
                Y.append(i)

            dnn = ml.DNNModel(args['-s'])
            dnn.cross(np.array(X), np.array(Y), int(args['--e']))
            dnn.train(np.array(X), np.array(Y), int(args['--e']))
            print(len(X), len(y_test_2))
            dnn.save()
            dnn.load()
            dnn.evaluate(np.array(X_test_2), np.array(y_test_2))
            prediction = dnn.predict(np.array(X_test_2))
            pred = translate_pred(prediction)
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv3(all_valid_gt, pred, indicies_test2)
            rp.report_pdf3(all_hits, all_samples, all_valid_gt, pred, indicies_test2)

        else:
            indicies = []
            for i in range(len(X)):
                indicies.append(i)
            X_train, X_test, y_train, y_test, indicies_train, indicies_test = train_test_split(X, Y, indicies, test_size=0.20, random_state=100)
            dnn = ml.DNNModel(args['-s'])
            dnn.cross(np.array(X_train), np.array(y_train), int(args['--e']))
            dnn.train(np.array(X_train), np.array(y_train), int(args['--e']))
            dnn.save()
            dnn.load()
            print(len(X_train), len(X_test))
            dnn.evaluate(np.array(X_test), np.array(y_test))
            prediction = dnn.predict(np.array(X_test))
            pred = translate_pred(prediction)
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv3(all_valid_gt, pred, indicies_test)
            rp.report_pdf3(all_hits, all_samples, all_valid_gt, pred, indicies_test)

    elif args['tts_rf_features']:
        import DT_Model as dt
        parser = dp.DataParser(args['-d'], dp.Mode.TRAIN)
        all_data = parser.parseData()
        all_names = []
        all_hits = []
        all_samples = []
        all_valid_gt = []
        df = pd.DataFrame(
            columns=['Name', 'Confidence', 'Score', '#TOP Hit Ions in Compound', '#Top Compound Ions in Hit',
                     'Molecular Ion in Compound', 'Correlation Percentage'])
        table_creator = ft.gt_table()
        for data in all_data:
            sample_name = data['name']
            hits = data['hits']
            sample = data['sample']
            ground_truth = data['ground_truth']
            valid_gt = get_valid_gt(ground_truth, sample)
            all_names.append(sample_name)
            all_hits.append(hits)
            all_samples.append(sample)
            all_valid_gt.append(valid_gt)
            for (peak_num, name, confidence, molecular_ion) in valid_gt:
                s, x, y, z, m = table_creator.table(hits[name], sample[peak_num - 1], molecular_ion)
                df.loc[-1] = [str(name), confidence, s, x, y, z, m]
                df.index = df.index + 1
                df = df.sort_index()
        df = df.iloc[::-1]
        df = df.drop(['Name'], axis=1)

        if args['--w']:
            parser2 = dp.DataParser(args['--w'], dp.Mode.TRAIN)
            all_data = parser2.parseData()
            all_names = []
            all_hits = []
            all_samples = []
            all_valid_gt = []
            df2 = pd.DataFrame(
                columns=['Name', 'Confidence', 'Score', '#TOP Hit Ions in Compound', '#Top Compound Ions in Hit',
                         'Molecular Ion in Compound', 'Correlation Percentage'])
            table_creator = ft.gt_table()
            for data in all_data:
                sample_name = data['name']
                hits = data['hits']
                sample = data['sample']
                ground_truth = data['ground_truth']
                valid_gt = get_valid_gt(ground_truth, sample)
                all_names.append(sample_name)
                all_hits.append(hits)
                all_samples.append(sample)
                all_valid_gt.append(valid_gt)
                for (peak_num, name, confidence, molecular_ion) in valid_gt:
                    s, x, y, z, m = table_creator.table(hits[name], sample[peak_num - 1], molecular_ion)
                    df2.loc[-1] = [str(name), confidence, s, x, y, z, m]
                    df2.index = df2.index + 1
                    df2 = df2.sort_index()
            df2 = df2.iloc[::-1]
            df2 = df2.drop(['Name'], axis=1)

            indicies = []
            for i in range(len(df2['Confidence'])):
                indicies.append(i)
            X_train2, X_test2, indicies_train, indicies_test = train_test_split(df2, indicies, test_size=0.20, random_state=100)

            final_df = pd.concat([df, X_train2], ignore_index=True)

            X_train = final_df.drop(['Confidence'], axis = 1)
            y_train = final_df['Confidence']

            X_test = X_test2.drop(['Confidence'], axis = 1)
            y_test = X_test2['Confidence']

            y_train = y_train.astype('int')
            y_test = y_test.astype('int')

            print(len(X_train))
            print(len(X_test))

            model = dt.model_train(X_train, y_train, args['-s'])
            pred = dt.model_test(X_test, y_test, args['-s'])
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv3(all_valid_gt, pred, indicies_test)
            rp.report_pdf3(all_hits, all_samples, all_valid_gt, pred, indicies_test)

        else:
            indicies = []
            for i in range(len(df['Confidence'])):
                indicies.append(i)
            X_train, X_test, y_train, y_test, indicies_train, indicies_test = train_test_split(df.drop(['Confidence'], axis = 1), df['Confidence'], indicies, test_size=0.20, random_state=100)
            y_train = y_train.astype('int')
            y_test = y_test.astype('int')
            print(len(y_train))
            print(len(y_test))
            model = dt.model_train(X_train, y_train, args['-s'])
            pred = dt.model_test(X_test, y_test, args['-s'])
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv3(all_valid_gt, pred, indicies_test)
            rp.report_pdf3(all_hits, all_samples, all_valid_gt, pred, indicies_test)
