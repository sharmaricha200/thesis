"""
Usage:
  run.py ml (train|test|predict) -d=<data_dir> -s=<model_save_file_path> [--e=<epochs>]
  run.py algo -d=<data_dir> (test|predict) [--st=<similarity_threshold>] [--pt=<percent_threshold>]
  run.py -h

Options:
  -h --help                     Show this screen.
  -d=<data_dir>                 Data directory path.
  --st=<similarity_threshold>   Similarity threshold [default: 600].
  --pt=<percent threshold>      Percent threshold [default: 80].
  --e=<epochs>                  Number of epochs [default: 30]
  -s=<model_save_file_path>     File path where model is saved
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

def get_valid_gt(in_gt, sample):
    out_gt = []
    for (peak_num, name, confidence) in in_gt:
        if name in hits and sample[peak_num - 1]['name'] == name:
            out_gt.append((peak_num, name, confidence))
    return out_gt

def translate_pred(pred):
    ret  = []
    pred[pred <= 0.5] = 0
    pred[pred > 0.5] = 1
    for p in pred:
        if (p[0] == 0 and p[1] == 0 and p[2] == 1):
            ret.append(2)
        elif (p[0] == 0 and p[1] == 1 and p[2] == 0):
            ret.append(1)
        elif (p[0] == 1 and p[1] == 0 and p[2] == 0):
            ret.append(0)
        else:
            ret.append(-1)
    return ret

def get_stats(pred, gt, positive):
    p = np.where(gt == positive)
    c = pred[p]
    tp = np.sum(c == positive)
    fn = np.sum(c != positive)
    p1 = np.where(pred == positive)
    c1 = gt[p1]
    fp = np.sum(c1 != positive)
    tpr = tp * 100 / (tp + fn)
    ppv = tp * 100 / (tp + fp)
    return tpr, ppv

if __name__ == '__main__':
    args = docopt(__doc__, version='1.0')
    #print(args)
    if args['ml']:
        import MLModel as ml
        if args['train']:
            parser = dp.DataParser(args['-d'], dp.Mode.TRAIN)
            all_data = parser.parseData()
            X = []
            Y = []
            for data in all_data:
                sample_name = data['name']
                hits = data['hits']
                sample = data['sample']
                ground_truth = data['ground_truth']
                valid_gt = get_valid_gt(ground_truth, sample)
                for (peak_num, name, confidence) in valid_gt:
                    if confidence == 2:
                        gt = [0, 0, 1]
                    elif confidence == 1:
                        gt = [0, 1, 0]
                    elif confidence == 0:
                        gt = [1, 0, 0]
                    sample_spectrum = sample[peak_num - 1]['spectrum']
                    sample_spectrum = sample_spectrum/np.linalg.norm(sample_spectrum)
                    hit_spectrum = hits[name]['spectrum']
                    hit_spectrum = hit_spectrum/np.linalg.norm(hit_spectrum)
                    X.append(np.concatenate((sample_spectrum, hit_spectrum)))
                    Y.append(gt)
            dnn = ml.DNNModel(args['-s'])
            dnn.train(np.array(X), np.array(Y), int(args['--e']))
            dnn.save()
        elif args['test']:
            parser = dp.DataParser(args['-d'], dp.Mode.TEST)
            data = parser.parseData()
            sample_name = data['name']
            hits = data['hits']
            sample = data['sample']
            ground_truth = data['ground_truth']
            valid_gt = get_valid_gt(ground_truth, sample)
            X = []
            Y = []
            for (peak_num, name, confidence) in valid_gt:
                if confidence == 2:
                    gt = [0, 0, 1]
                elif confidence == 1:
                    gt = [0, 1, 0]
                elif confidence == 0:
                    gt = [1, 0, 0]
                sample_spectrum = sample[peak_num - 1]['spectrum']
                sample_spectrum = sample_spectrum / np.linalg.norm(sample_spectrum)
                hit_spectrum = hits[name]['spectrum']
                hit_spectrum = hit_spectrum / np.linalg.norm(hit_spectrum)
                X.append(np.concatenate((sample_spectrum, hit_spectrum)))
                Y.append(gt)
            dnn = ml.DNNModel(args['-s'])
            dnn.load()
            dnn.evaluate(np.array(X), np.array(Y))
            prediction = dnn.predict(np.array(X))
            pred = translate_pred(prediction)
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv(sample_name, valid_gt, pred)
            rp.report_pdf(sample_name, hits, sample, valid_gt, pred)
        elif args['predict']:
            parser = dp.DataParser(args['-d'], dp.Mode.PREDICT)
            data = parser.parseData()
            sample_name = data['name']
            hits = data['hits']
            sample = data['sample']
            peak_num = 0
            pred = []
            X = []
            for compound in sample:
                peak_num = peak_num + 1
                name = compound['name']
                if name in hits:
                    sample_spectrum = compound['spectrum']
                    sample_spectrum = sample_spectrum / np.linalg.norm(sample_spectrum)
                    hit_spectrum = hits[name]['spectrum']
                    hit_spectrum = hit_spectrum / np.linalg.norm(hit_spectrum)
                    X.append(np.concatenate((sample_spectrum, hit_spectrum)))
                    pred.append((peak_num, name, 0))
            dnn = ml.DNNModel(args['-s'])
            dnn.load()
            prediction = dnn.predict(np.array(X))
            pred_arr = translate_pred(prediction)
            i = 0
            for Y in pred_arr:
                pred[i] = (pred[i][0], pred[i][1], Y)
                i = i + 1
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'ml')
            rp.report_csv1(sample_name, pred)
            rp.report_pdf1(sample_name, hits, sample, pred)
    elif args['algo']:
        import AlgoModel as am
        model = am.AlgoModel(int(args['--st']), int(args['--pt']))
        if args['test']:
            parser = dp.DataParser(args['-d'], dp.Mode.TEST)
            data = parser.parseData()
            pred = []
            gt_conf = []
            sample_name = data['name']
            hits = data['hits']
            sample = data['sample']
            ground_truth = data['ground_truth']
            valid_gt = get_valid_gt(ground_truth, sample)
            for (peak_num, name, confidence) in valid_gt:
                conf, percentage_lib_hit_present, molecular_ion_hit, top_three_ion_list_sample = model.predict(hits[name], sample[peak_num - 1])
                pred.append(conf)
                gt_conf.append(confidence)
            pred = np.array(pred)
            gt_conf = np.array(gt_conf)
            sensitivity, precision = get_stats(pred, gt_conf, 2)
            print("For High confidence, Sensitivity: {sens}%, Precision: {prec}%".
                  format(sens=sensitivity, prec=precision))
            sensitivity, precision = get_stats(pred, gt_conf, 0)
            print("For Low confidence, Sensitivity: {sens}%, Precision: {prec}%".
                  format(sens=sensitivity, prec=precision))
            sensitivity, precision = get_stats(pred, gt_conf, 1)
            print("For Medium confidence, Sensitivity: {sens}%, Precision: {prec}%".
                  format(sens=sensitivity, prec=precision))
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'algo')
            rp.report_csv(sample_name, valid_gt, pred)
            rp.report_pdf(sample_name, hits, sample, valid_gt, pred)
        elif args['predict']:
            parser = dp.DataParser(args['-d'], dp.Mode.PREDICT)
            data = parser.parseData()
            peak_num = 0
            pred = []
            sample_name = data['name']
            hits = data['hits']
            sample = data['sample']
            for compound in sample:
                peak_num = peak_num + 1
                name = compound['name']
                if name in hits:
                    conf, percentage_lib_hit_present, molecular_ion_hit, top_three_ion_list_sample = model.predict(hits[name], compound)
                    pred.append((peak_num, name, conf))
            rp = rg.ReportGenerator(ROOT_PATH + "/report", 'algo')
            rp.report_csv1(sample_name, pred)
            rp.report_pdf1(sample_name, hits, sample, pred)
