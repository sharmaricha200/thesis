import re
import collections
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "/Users/vinay/thesis/data/sample2"
MAX_X = 801

def read_sample_file(file_name):
    sample_data = open(file_name).read()
    count_sample = sample_data.count('NAME:')
    print(count_sample)


read_sample_file("/Users/vinay/thesis/data/peak_true_sample.MSP")


def read_msp_file(file_name, is_sample):
    try:
        msp = open(file_name, "r")
    except Exception as err:
        print("Error")
        raise err
    else:
        msp_line_list = msp.readlines()
        msp.close()
        mz_data = {}

        mz_data['spectrum'] = np.zeros((MAX_X,))
        if is_sample:
            mz_data['name'] = ""
            mz_data['score'] = 0
        else:
            name_re = re.compile('NAME: Library Hit - similarity ([0-9]*), \"(.*)\"')
            name_match = name_re.match(msp_line_list[0])
            if name_match is None:
                print('File format error')
                raise Exception
            mz_data['score'] = int(name_match.group(1))
            mz_data['name'] = name_match.group(2)

        peaks_re = re.compile('Num Peaks: ([0-9]*)')
        peaks_match = peaks_re.match(msp_line_list[1])
        if peaks_match is None:
            print('File format error')
            # TODO: Make sure this exception is handled
            raise Exception
        else:
            mz_data['peaks'] = int(peaks_match.group(1))
            for line in msp_line_list[2:]:
                for pair in line.split(';'):
                    data = pair.split()
                    if data:
                        index = int(data[0])
                        value = int(data[1])
                        mz_data['spectrum'][index] = value
            return mz_data


def get_top_three(mz_intensity):
    ind = np.argpartition(mz_intensity, -3)[-3:]
    return ind[np.argsort(mz_intensity[ind])]

def get_molecular_ion(mz_intensity):
    return np.argmax(mz_intensity)

def get_percentage_lib_hit(mz_intensity_compd, mz_intensity_lib):
    mz_compd_list = np.where(mz_intensity_compd != 0)
    mz_lib_list = np.where(mz_intensity_lib != 0)
    compd_in_lib = np.isin(mz_compd_list, mz_lib_list).sum()
    lib_in_compd = np.isin(mz_lib_list, mz_compd_list).sum()
    percentage_compd_in_lib_hit_present = compd_in_lib * 100 / len(mz_compd_list)
    percentage_lib_hit_in_compd_present = lib_in_compd * 100 / len(mz_lib_list)
    print(percentage_compd_in_lib_hit_present)
    return percentage_lib_hit_in_compd_present

def get_classification_compd(compd_file, lib_file):
    try:
        mz_data_compd = read_msp_file(compd_file, True)
        mz_data_lib = read_msp_file(lib_file, False)
    except Exception as err:
        print("Error in processing!!!! {0}".format(err))
    else:
        similarity_score = mz_data_lib['score']
        name_compd = mz_data_lib['name']
        if similarity_score >= 600:
            mz_intensity_compd = mz_data_compd['spectrum']
            mz_intensity_lib = mz_data_lib['spectrum']
            top_three_ion_list_compd = get_top_three(mz_intensity_compd)
            top_three_ion_list_lib = get_top_three(mz_intensity_lib)
            mz_lib_list = np.where(mz_intensity_lib != 0)
            mz_compd_list = np.where(mz_intensity_compd != 0)
            molecular_ion_lib = get_molecular_ion(mz_intensity_lib)
            percentage_lib_hit_present = get_percentage_lib_hit(mz_intensity_compd, mz_intensity_lib)
            if np.isin(top_three_ion_list_lib, mz_compd_list).sum() >= 3:
                if np.isin(molecular_ion_lib, mz_compd_list).sum() > 0:
                    if percentage_lib_hit_present >= 80:
                        print("Prediction is : High Confidence")
                    else:
                        print("Prediction is : Medium Confidence")
                else:
                    print("Prediction is : Medium Confidence")

            else:
                print("Prediction is : Low Confidence")

        else:
            print("Prediction is : Low Confidence")
        print("Name of the compound is:", name_compd)
        print("The similarity score is :", similarity_score)
        print("The top three ions in the compound are:", top_three_ion_list_compd)
        print("The molecular ion of the compound is:", molecular_ion_lib)
        print("The percentage of ions of the library hit present in the compound", percentage_lib_hit_present, "%")
        plt.plot(mz_intensity_compd)
        plt.plot(-mz_intensity_lib)
        plt.show()
        return similarity_score, top_three_ion_list_compd, molecular_ion_lib, percentage_lib_hit_present


print(get_classification_compd("/Users/vinay/Desktop/Phthalic acid, monocyclohexyl ester Peak True.msp",
                               "/Users/vinay/Desktop/Phthalic acid, monocyclohexyl ester Library.msp"))
print(get_classification_compd("/Users/vinay/Desktop/2-(p-Tolyl)ethylamine Peak True.msp",
                               "/Users/vinay/Desktop/2-(p-Tolyl)ethylamine Library.msp"))
