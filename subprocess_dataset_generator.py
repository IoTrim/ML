from datasetGenerator import datasetGenerator
from multiprocessing import Pool

def gen(i):
    dG = datasetGenerator(i).loadDevicesFromDB().readPcaps().createDataset().createDataframe()

rng = [i*2 for i in range(25, 250)]
print(rng)
with Pool(processes=25) as pool:
    pool.map(gen, rng)