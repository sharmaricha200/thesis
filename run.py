"""
Usage:
  run.py ml (train|test) -d=<data_dir> -s=<save_file_path> [--e=<epochs>]
  run.py algo -d=<data_dir> [--st=<similarity_threshold>] [--pt=<percent_threshold>]
  run.py -h

Options:
  -h --help                     Show this screen.
  -d=<data_dir>                 Data directory path.
  --st=<similarity_threshold>   Similarity threshold [default: 600].
  --pt=<percent threshold>      Percent threshold [default: 80].
  --e=<epochs>                  Number of epochs [default: 30]
  -s=<save_file_path>           File path where model is saved
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
        if (p[0] == 1 and p[1] == 0):
            ret.append(1)
        elif (p[0] == 0 and p[1] == 1):
            ret.append(2)
        else:
            ret.append(-1)
    return ret

if __name__ == '__main__':
    args = docopt(__doc__, version='1.0')
    #print(args)
    parser = dp.DataParser(args['-d'])
    data = parser.parseData()
    hits = data[0]['hits']
    sample = data[0]['sample']
    rp = rg.ReportGenerator(ROOT_PATH + "/report", 'algo')
    i = 0
    m = 1
    if args['ml']:
        import MLModel as ml
        dnn = ml.DNNModel(args['-s'])
        if args['train']:
            ground_truth = data[0]['truth_train']
        elif args['test']:
            ground_truth = data[0]['truth_test']

        valid_gt = get_valid_gt(ground_truth, sample)
        dt = np.empty([len(valid_gt), 1602])
        gt = np.empty([len(valid_gt), 2])

        for (peak_num, name, confidence) in valid_gt:
            dt[i] = np.concatenate((sample[peak_num - 1]['spectrum'], hits[name]['spectrum']))
            if confidence == 2:
                gt[i][0] = 0
                gt[i][1] = 1
            elif confidence == 1:
                gt[i][0] = 1
                gt[i][1] = 0
            i = i + 1

        x = dt[:i]
        y = gt[:i]
        if args['train']:
            dnn.train(x, y, int(args['--e']))
        elif args['test']:
            dnn.load()
            dnn.evaluate(x, y)
            prediction = dnn.predict(x)
            pred = translate_pred(prediction)
            i = 0
            for (peak_num, name, confidence) in valid_gt:
                print("{peak_num};{name};{confidence};{pred}".format(peak_num=peak_num, name=name, confidence=confidence,pred=pred[i]))
                i = i + 1

    elif args['algo']:
        import AlgoModel as am
        model = am.AlgoModel(int(args['--st']), int(args['--pt']))
        pred = []
        gt_conf = []
        ground_truth = [*data[0]['truth_train'], *data[0]['truth_test']]
        valid_gt = get_valid_gt(ground_truth, sample)
        for (peak_num, name, confidence) in valid_gt:
            conf, percentage_lib_hit_present, molecular_ion_hit, top_three_ion_list_sample = model.predict(hits[name], sample[peak_num - 1])
            pred.append(conf)
            gt_conf.append(confidence)
        pred = np.array(pred)
        gt_conf = np.array(gt_conf)
        right = pred == gt_conf
        accuracy = np.sum(right) / right.size * 100
        print("acc = {acc}%".format(acc=accuracy))
        i = 0
        for (peak_num, name, confidence) in valid_gt:
            print("{peak_num};{name};{confidence};{pred}".format(peak_num=peak_num, name=name, confidence=confidence,pred=pred[i]))
            i = i+1
        print(pred)