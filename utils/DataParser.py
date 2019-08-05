import re
import collections
import numpy as np
import os
from os.path import isfile, join

MAX_X = 801
class DataParser:
    def __init__(self, rootDataDir):
        self.data = []
        self.ROOT_DATA_DIR = rootDataDir

    #parses all the files in given rootDataDir into a datastructure.
    #the rootDataDir is assumed to have following file organization:
    #rootDataDir
    #└── samplen
    #    ├── ground_truth.csv
    #    ├── hits
    #    │   ├── 1.msp
    #    │   ├── 2.msp
    #    │   ├── 3.msp
    #    │   └── ...
    #    └── peak_true.msp
    #Output is datastructure:
    #ret
    #   |
    #   ['hits']
    #   |      |__[name_0]
    #   |      |    |__['peaks']
    #   |      |    |__['spectrum']
    #   |      |    |__['score']
    #   |      |__[name_1]
    #   |      |    |__['peaks']
    #   |      |    |__['spectrum']
    #   |      |    |__['score']
    #   |      .
    #   |      .
    #   |      .
    #   |      |__[name_n]
    #   |           |__['peaks']
    #   |           |__['spectrum']
    #   |           |__['score']
    #   |
    #   ['sample']
    #   |        |__[0]
    #   |        |    |__['name']
    #   |        |    |__['peaks']
    #   |        |    |__['spectrum']
    #   |        |__[1]
    #   |        |    |__['name']
    #   |        |    |__['peaks']
    #   |        |    |__['spectrum']
    #   |        .
    #   |        .
    #   |        .
    #   |        |__[l]
    #   |             |__['name']
    #   |             |__['peaks']
    #   |             |__['spectrum']
    #   ['ground_truth']
    #           |__[0]
    #           |    |__(peak_num, name, confidence)
    #           |__[1]
    #           |    |__(peak_num, name, confidence)
    #           .
    #           .
    #           .
    #           |__[m]
    #                |__(peak_num, name, confidence)

    def parseData(self, test_ml=False):
        dirs = next(os.walk(self.ROOT_DATA_DIR))[1]
        for dir in dirs:
            if test_ml is True:
                if os.path.basename(dir) != 'test':
                    continue
            else:
                if os.path.basename(dir) == 'test':
                    continue
            currPath = os.path.join(self.ROOT_DATA_DIR, dir)
            hitsPath = os.path.join(currPath, "hits")
            sampleFile = os.path.join(currPath, "peak_true.msp")
            groundTruthFile = os.path.join(currPath, "ground_truth.csv")
            hitsFiles = next(os.walk(hitsPath))[2]
            hits = {}
            for hitsFile in hitsFiles:
                path = os.path.join(hitsPath, hitsFile)
                name, mzdata = self.__parseHitsFile(path)
                hits[name] = mzdata
            self.data.append({'name': os.path.basename(currPath),'hits': hits, 'sample':self.__parseSampleFile(sampleFile),
                              'ground_truth': self.__parseGroundTruth(groundTruthFile)})
        return self.data

    def __parseSampleFile(self, filename):
        name_re = re.compile('NAME: (.*)')
        peaks_re = re.compile('Num Peaks: ([0-9]*)')
        state = 0
        ret = []
        with open(filename, 'r', encoding="latin-1") as f:
            for l in f:
                line = l.strip()
                if state == 0:
                    mz_data = {}
                    name_match = name_re.match(line)
                    if name_match is None:
                        print('File format error on file: ' + filename)
                        raise Exception
                    name = name_match.group(1)
                    mz_data['name'] = name
                    state = 1
                    continue
                if state == 1:
                    peaks_match = peaks_re.match(line)
                    if peaks_match is None:
                        print('File format error on file: ' + filename)
                        # TODO: Make sure this exception is handled
                        raise Exception
                    else:
                        mz_data['peaks'] = int(peaks_match.group(1))
                    mz_data['spectrum'] = np.zeros((MAX_X,))
                    state = 2
                    continue
                if state == 2:
                    if not line:
                        ret.append(mz_data);
                        state = 0
                    else:
                        for pair in line.split(';'):
                            data = pair.split()
                            if data:
                                index = int(data[0])
                                value = int(data[1])
                                mz_data['spectrum'][index] = value
                    continue
        return ret

    def __parseHitsFile(self, filename):
        try:
            msp = open(filename, "r", encoding="latin-1")
        except Exception as err:
            print("Error opening file: " + filename)
            raise err
        else:
            msp_line_list = msp.readlines()
            msp.close()
            mz_data = {}

            mz_data['spectrum'] = np.zeros((MAX_X,))
            name_re = re.compile('NAME: Library Hit - similarity ([0-9]*), \"(.*)\"')
            name_match = name_re.match(msp_line_list[0])
            if name_match is None:
                print('File format error on file: ' + filename)
                raise Exception
            mz_data['score'] = int(name_match.group(1))
            name = name_match.group(2)

        peaks_re = re.compile('Num Peaks: ([0-9]*)')
        peaks_match = peaks_re.match(msp_line_list[1])
        if peaks_match is None:
            print('File format error on file: ' + filename)
            # TODO: Make sure this exception is handled
            raise Exception
        else:
            mz_data['peaks'] = int(peaks_match.group(1))
            for l in msp_line_list[2:]:
                line = l.strip()
                for pair in line.split(';'):
                    data = pair.split()
                    if data:
                        index = int(data[0])
                        value = int(data[1])
                        mz_data['spectrum'][index] = value
            return name, mz_data

    def __parseGroundTruth(self, filename):
        arr = np.genfromtxt(filename, delimiter=';', dtype=None, encoding='ascii')
        return arr
