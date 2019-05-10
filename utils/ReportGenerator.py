import os
import shutil

class ReportGenerator:
    def __init__(self, report_root, model_name):
        self.report_path = report_root + "/" + model_name
        if os.path.exists(self.report_path):
            shutil.rmtree(self.report_path)
        os.mkdir(self.report_path)
    def report(self, sample_name, name, percent, molecular_ion, top_three_ion):
        if not os.path.exists(self.report_path):
            os.mkdir(self.report_path)
        f = open(os.path.join(self.report_path, sample_name), "a+")
        f.write(name + "," + str(percent) + "\n")
        #f.close()
        pass
    def __del__(self):
        pass
