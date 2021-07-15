import re
import collections
import numpy as np
import os
from os.path import isfile, join
from enum import Enum

class Mode(Enum):
    TRAIN = 1
    TEST = 2
    PREDICT = 3

MAX_X = 1000
class DataParser:
    def __init__(self, rootDataDir, mode=Mode.TRAIN):
        self.ROOT_DATA_DIR = rootDataDir
        self.mode = mode

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

    def parseOneSample(self, currPath):
        hitsPath = os.path.join(currPath, "hits")
        sampleFile = os.path.join(currPath, "peak_true.msp")
        groundTruthFile = os.path.join(currPath, "ground_truth.tsv")
        compoundsFile = os.path.join(currPath, "compounds.tsv")
        hitsFiles = next(os.walk(hitsPath))[2]
        hits = {}
        for hitsFile in hitsFiles:
            path = os.path.join(hitsPath, hitsFile)
            name, mzdata = self.__parseHitsFile(path)
            hits[name] = mzdata
        if self.mode == Mode.TRAIN or self.mode == Mode.TEST:
            return {'name': os.path.basename(currPath), 'hits': hits, 'sample': self.__parseSampleFile(sampleFile),
                    'ground_truth': self.__parseGroundTruth(groundTruthFile)}
        else:
            return {'name': os.path.basename(currPath), 'hits': hits, 'sample': self.__parseSampleFile(sampleFile),
                    'compounds': self._parseCompounds(compoundsFile)}

    def parseData(self):
        #if self.mode == Mode.TRAIN:
        data = []
        dirs = next(os.walk(self.ROOT_DATA_DIR))[1]
        for dir_name in dirs:
            curr_path = os.path.join(self.ROOT_DATA_DIR, dir_name)
            data.append(self.parseOneSample(curr_path))
        return data
        # elif self.mode == Mode.TEST or self.mode == Mode.PREDICT:
        #     return self.parseOneSample(self.ROOT_DATA_DIR)

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
                                mz_data['spectrum'][index - 1] = value
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
                        mz_data['spectrum'][index - 1] = value
            return name, mz_data

    def __parseGroundTruth(self, filename):
        arr = np.genfromtxt(filename, delimiter='\t', dtype="i8,|U128,i8,i8", encoding='utf-8')
        return arr

    def _parseCompounds(self, filename):
        arr = np.genfromtxt(filename, delimiter='\t', dtype = "i8,|U128,i8", encoding='utf-8')
        return arr