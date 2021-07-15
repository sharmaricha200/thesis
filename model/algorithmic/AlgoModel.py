import pandas as pd
import numpy as np
import math

NOISE_THRESHOLD = 0.02
PERCENT_HIGH = 80
TOP_N = 3
class AlgoModel:
    def __init__(self, similarity_th, percent_th):
        self.SIMILARITY_THRESHOLD = similarity_th
        self.PERCENT_THRESHOLD = percent_th
       # print("Configuration: {phigh}%, {pmed}%, {noise}, {topn}".
       #       format(phigh=PERCENT_HIGH, pmed=PERCENT_MED, noise=NOISE_THRESHOLD, topn=TOP_N))
        pass

    def __get_top_n(self, mz_intensity):
        ind = np.argpartition(mz_intensity, -TOP_N)[-TOP_N:]
        return ind[np.argsort(mz_intensity[ind])]

    def __get_molecular_ion(self, mz_intensity):
        return mz_intensity[0][-1]

    def __get_percentage_lib_hit(self, mz_intensity_compd, mz_intensity_lib):
        mz_compd_list = np.where(mz_intensity_compd > NOISE_THRESHOLD)
        mz_lib_list = np.where(mz_intensity_lib > NOISE_THRESHOLD)
        compd_in_lib = np.isin(mz_compd_list, mz_lib_list).sum()
        lib_in_compd = np.isin(mz_lib_list, mz_compd_list).sum()
        percentage_compd_in_lib_hit_present = compd_in_lib * 100 / len(mz_compd_list[0])
        percentage_lib_hit_in_compd_present = lib_in_compd * 100 / len(mz_lib_list[0])
        return max(percentage_compd_in_lib_hit_present, percentage_lib_hit_in_compd_present)

    def predict(self, hit, sample, molecular_ion):
        score = hit['score']
        confidence = 0
        percentage_lib_hit_present = 0
        molecular_ion_hit = 0
        top_n_ion_list_sample = []
        if score >= self.SIMILARITY_THRESHOLD:
            sample_spectrum = sample['spectrum']
            hit_spectrum = hit['spectrum']
            sample_spectrum = sample_spectrum / np.linalg.norm(sample_spectrum)
            hit_spectrum = hit_spectrum/np.linalg.norm(hit_spectrum)
            hit_spectrum[0:49] = 0
            top_n_ion_list_sample = self.__get_top_n(sample_spectrum)
            top_n_ion_list_hit = self.__get_top_n(hit_spectrum)
            hit_list = np.where(hit_spectrum >= NOISE_THRESHOLD)
            sample_list = np.where(sample_spectrum >= NOISE_THRESHOLD)
            percentage_lib_hit_present = self.__get_percentage_lib_hit(sample_spectrum, hit_spectrum)
            dot_product = np.dot(sample_spectrum, hit_spectrum) * 100
            if (np.isin(top_n_ion_list_hit, sample_list).sum() >= TOP_N
                and np.isin(molecular_ion-1, sample_list).sum() > 0
                and np.isin(top_n_ion_list_sample, hit_list).sum() >= TOP_N
                and dot_product >= PERCENT_HIGH):
                confidence = 2
            else:
                confidence = 0

        return confidence, percentage_lib_hit_present, molecular_ion - 1, top_n_ion_list_sample