import numpy as np

class AlgoModel:
    def __init__(self, similarity_th, percent_th):
        self.SIMILARITY_THRESHOLD = similarity_th
        self.PERCENT_THRESHOLD = percent_th
        pass

    def __get_top_three(self, mz_intensity):
        ind = np.argpartition(mz_intensity, -3)[-3:]
        return ind[np.argsort(mz_intensity[ind])]

    def __get_molecular_ion(self, mz_intensity):
        return mz_intensity[0][-1]

    def __get_percentage_lib_hit(self, mz_intensity_compd, mz_intensity_lib):
        mz_compd_list = np.where(mz_intensity_compd >= 20)
        mz_lib_list = np.where(mz_intensity_lib >= 20)
        compd_in_lib = np.isin(mz_compd_list, mz_lib_list).sum()
        lib_in_compd = np.isin(mz_lib_list, mz_compd_list).sum()
        percentage_compd_in_lib_hit_present = compd_in_lib * 100 / len(mz_compd_list[0])
        percentage_lib_hit_in_compd_present = lib_in_compd * 100 / len(mz_lib_list[0])
        #print(percentage_lib_hit_in_compd_present)
        #return percentage_lib_hit_in_compd_present
        return max(percentage_compd_in_lib_hit_present, percentage_lib_hit_in_compd_present)

    def predict(self, hit, sample):
        score = hit['score']
        confidence = 0
        if score >= self.SIMILARITY_THRESHOLD:
            sample_spectrum = sample['spectrum']
            hit_spectrum = hit['spectrum']
            hit_spectrum[0:50] = 0
            top_three_ion_list_sample = self.__get_top_three(sample_spectrum)
            top_three_ion_list_hit = self.__get_top_three(hit_spectrum)
            hit_list = np.where(hit_spectrum >= 20)
            sample_list = np.where(sample_spectrum >= 20)
            molecular_ion_hit = self.__get_molecular_ion(hit_list)
            molecular_ion_sample = self.__get_molecular_ion(sample_list)
            percentage_lib_hit_present = self.__get_percentage_lib_hit(sample_spectrum, hit_spectrum)
            if np.isin(top_three_ion_list_hit, sample_list).sum() >= 2:
                if np.isin(molecular_ion_hit, sample_list).sum() > 0:
                    if percentage_lib_hit_present >= self.PERCENT_THRESHOLD:
                        #print("Prediction is : High Confidence")
                        confidence = 2
                    else:
                        confidence = 1
                        #print("Prediction is : Medium Confidence")
                else:
                    confidence = 1
                    #print("Prediction is : Medium Confidence")

        else:
            confidence = 0
            #print("Prediction is : Low Confidence")

        return confidence, percentage_lib_hit_present, molecular_ion_hit, top_three_ion_list_sample
        #print("Name of the compound is:", name)
        #print("The similarity score is :", score)
        #print("The top three ions in the compound are:", top_three_ion_list_sample)
        #print("The molecular ion of the compound is:", molecular_ion_hit)
        #print("The percentage of ions of the library hit present in the compound", percentage_lib_hit_present, "%")