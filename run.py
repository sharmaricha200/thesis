import sys
ROOT_PATH = "/Users/vinay/thesis"

sys.path.insert(0, ROOT_PATH + "/model/algorithmic")
sys.path.insert(0, ROOT_PATH + "/utils")

import DataParser as dp
import AlgoModel as am
import ReportGenerator as rg


parser = dp.DataParser(ROOT_PATH + "/data/")
data = parser.parseData()
hits = data[0]['hits']
sample = data[0]['sample']
model = am.AlgoModel(600, 80)
rp = rg.ReportGenerator(ROOT_PATH + "/report", 'algo')

for s in sample:
    name = s['name']
    if name in hits:
        h = hits[name]
        percent, molecular_ion, top_three_ion = model.predict(name, h, s)
        rp.report("sample1", name, percent, molecular_ion, top_three_ion)
    else:
        print("This compound is not present in lib hits. Please check input files. Compound: " + name)