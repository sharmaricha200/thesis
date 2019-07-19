"""
Usage:
  run.py ml train -d=<data_dir> -e=<epochs>
  run.py ml test -d=<data_dir>
  run.py algo -d=<data_dir> [--st=<similarity_threshold>] [--pt=<percent_threshold>]
  run.py -h

Options:
  -h --help                     Show this screen.
  -d=<data_dir>                 Data directory path.
  --st=<similarity_threshold>   Similarity threshold [default: 600].
  --pt=<percent threshold>      Percent threshold [default: 80].
  -e=<epochs>                   Number of epochs [default: ]
"""

from docopt import docopt
import sys
import os

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
    args = docopt(__doc__, version='Thesis')
    print(args)
    parser = dp.DataParser(args['-d'])
    data = parser.parseData()
    hits = data[0]['hits']
    sample = data[0]['sample']
    ground_truth = data[0]['truth']
    np.random.shuffle(ground_truth)
    rp = rg.ReportGenerator(ROOT_PATH + "/report", 'algo')
    i = 0
    m = 1
    if args['ml']:
        dnn = ml.DNNModel()
        train = np.empty([100, 1602])
        gt = np.empty([100, 2])
        for (peak_num, name, confidence) in ground_truth:
            if name in hits and sample[peak_num - 1]['name'] == name:
                #print("Name: " + name)
                train[i] = np.concatenate((sample[peak_num - 1]['spectrum'], hits[name]['spectrum']))
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
        test = train[25:32]
        test_y = gt[25:32]
        train = train[0:25]
        gt = gt[0:25]
        dnn.train(train, gt, int(args['-e']))
        pred = dnn.predict(test)
        print(pred)
        pred[pred <= 0.5] = 0
        pred[pred > 0.5] = 1
        print("Test--->")
        print(test_y)
        dnn.evaluate(test, test_y)
        dnn.save()
    elif args['algo']:
        model = am.AlgoModel(int(args['--st']), int(args['--pt']))
        pred = []
        gt_conf = []
        for (peak_num, name, confidence) in ground_truth:
            if name in hits and sample[peak_num - 1]['name'] == name:
                conf, percentage_lib_hit_present, molecular_ion_hit, top_three_ion_list_sample = model.predict(hits[name], sample[peak_num - 1])
                pred.append(conf)
                gt_conf.append(confidence)
        pred = np.array(pred)
        gt_conf = np.array(gt_conf)
        print(gt_conf)
        hehe = pred == gt_conf
        print(np.sum(hehe) / hehe.size * 100)
    print("pred --->")
    print(pred)


