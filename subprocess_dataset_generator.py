from datasetGenerator import datasetGenerator
from multiprocessing import Pool

def gen(i):
    dG = datasetGenerator(i).loadDevicesFromDB().readPcaps().createDataset().createDataframe()

rng = [12, 14, 18, 30, 50, 60, 70, 80, 90, 150,200, 250, 300, 350, 400, 500]
with Pool(processes=len(rng)) as pool:
    pool.map(gen, rng)