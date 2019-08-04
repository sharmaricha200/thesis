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
import AlgoModel as am
import ReportGenerator as rg
import MLModel as ml
import numpy as np

#for s in sample:
#    name = s['name']
#    if name in hits:
#        h = hits[name]
#        percent, molecular_ion, top_three_ion = model.predict(name, h, s)
#        rp.report("sample1", name, percent, molecular_ion, top_three_ion)
#    else:
#        print("This compound is not present in lib hits. Please check input files. Compound: " + name)


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
        dnn = ml.DNNModel(args['-s'])
        if args['train']:
            ground_truth = data[0]['truth_train']
        elif args['test']:
            ground_truth = data[0]['truth_test']

        dt = np.empty([len(ground_truth), 1602])
        gt = np.empty([len(ground_truth), 2])
        for (peak_num, name, confidence) in ground_truth:
            if name in hits and sample[peak_num - 1]['name'] == name:
                dt[i] = np.concatenate((sample[peak_num - 1]['spectrum'], hits[name]['spectrum']))
                if confidence == 2:
                    gt[i][0] = 0
                    gt[i][1] = 1
                elif confidence == 1:
                    gt[i][0] = 1
                    gt[i][1] = 0
                i = i + 1
            else:
                #print("Name not in hits: " + name)
                pass
        x = dt[:i]
        y = gt[:i]
        if args['train']:
            dnn.train(x, y, int(args['--e']))
        elif args['test']:
            dnn.load()
            dnn.evaluate(x, y)
            pred = dnn.predict(x)
            pred[pred <= 0.5] = 0
            pred[pred > 0.5] = 1
            print("Test--->")
            print(y)
            print("Prediction--->")
            print(pred)

    elif args['algo']:
        model = am.AlgoModel(int(args['--st']), int(args['--pt']))
        pred = []
        gt_conf = []
        ground_truth = data[0]['truth_train']
        for (peak_num, name, confidence) in ground_truth:
            if name in hits and sample[peak_num - 1]['name'] == name:
                conf, percentage_lib_hit_present, molecular_ion_hit, top_three_ion_list_sample = model.predict(hits[name], sample[peak_num - 1])
                pred.append(conf)
                gt_conf.append(confidence)
        ground_truth = data[0]['truth_test']
        for (peak_num, name, confidence) in ground_truth:
            if name in hits and sample[peak_num - 1]['name'] == name:
                conf, percentage_lib_hit_present, molecular_ion_hit, top_three_ion_list_sample = model.predict(hits[name], sample[peak_num - 1])
                pred.append(conf)
                gt_conf.append(confidence)
        pred = np.array(pred)
        gt_conf = np.array(gt_conf)
        print("Test -->")
        print(gt_conf)
        right = pred == gt_conf
        print(np.sum(right) / right.size * 100)
        print("Prediction --->")
        print(pred)


