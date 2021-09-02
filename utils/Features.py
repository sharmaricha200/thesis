import numpy as np

NOISE_THRESHOLD = 0.02
TOP_N = 3

class gt_table:
    def __init__(self):
        pass

    def __get_top_n(self, mz_intensity):
        ind = np.argpartition(mz_intensity, -TOP_N)[-TOP_N:]
        return ind[np.argsort(mz_intensity[ind])]

    def __get_molecular_ion(self, mz_intensity):
        return mz_intensity[0][-1]

    def __get_percentage_lib_hit(self, mz_intensity_compd, mz_intensity_lib):
        mz_compd_list = np.where(mz_intensity_compd >= NOISE_THRESHOLD)
        mz_lib_list = np.where(mz_intensity_lib >= NOISE_THRESHOLD)
        compd_in_lib = np.isin(mz_compd_list, mz_lib_list).sum()
        lib_in_compd = np.isin(mz_lib_list, mz_compd_list).sum()
        percentage_compd_in_lib_hit_present = compd_in_lib * 100 / len(mz_compd_list[0])
        percentage_lib_hit_in_compd_present = lib_in_compd * 100 / len(mz_lib_list[0])
        return max(percentage_compd_in_lib_hit_present, percentage_lib_hit_in_compd_present)

    def table(self, hit, sample, molecular_ion):
        score = hit['score']
        sample_spectrum = sample['spectrum']
        hit_spectrum = hit['spectrum']
        sample_spectrum = sample_spectrum / np.linalg.norm(sample_spectrum)
        hit_spectrum = hit_spectrum/np.linalg.norm(hit_spectrum)
        hit_spectrum[0:49] = 0
        top_n_ion_list_sample = self.__get_top_n(sample_spectrum)
        top_n_ion_list_hit = self.__get_top_n(hit_spectrum)
        hit_list = np.where(hit_spectrum >= NOISE_THRESHOLD)
        sample_list = np.where(sample_spectrum >= NOISE_THRESHOLD)
        dot_product = np.dot(sample_spectrum, hit_spectrum) * 100
        
        hit_in_sample = np.isin(top_n_ion_list_hit, sample_list).sum()
        sample_in_hit = np.isin(top_n_ion_list_sample, hit_list).sum()
        percentage = dot_product
        mol_in_sample = np.isin(molecular_ion - 1, sample_list).sum()
        
        return score, hit_in_sample, sample_in_hit, mol_in_sample, percentage
