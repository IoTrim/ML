from NetworkTrainer import NetworkTrainer
import matplotlib.pyplot as plt
import numpy as np
import os

class NetworkCompare:

    def __init__(self, numNetworks):
        self.datasetDir = "datasets/"
        self.plotDir = "images/"
        self.loss = {}
        self.accuracy = {}
        self.seeds = [i**i for i in range(numNetworks)]

    def generateNetworks(self):
        files = os.listdir(self.datasetDir)
        suffix = "_s_dataset.csv"
        for f in sorted(files, key=lambda x : int(x[:-len(suffix)])):
            if f.endswith(suffix):
                num = f[:-len(suffix)]
                loss = []
                acc = []
                for i in self.seeds:
                    NN = NetworkTrainer(f,i)
                    NN.process_data()
                    NN.create_model()
                    NN.train_model()
                    l, a = NN.evaluate_model()
                    loss.append(l)
                    acc.append(a)

                self.loss[num] = np.mean(loss)
                self.accuracy[num] = np.mean(acc)

    def plot(self):
        d = {"Accuracy":self.accuracy, "Loss":self.loss}
        for k, v in d.items():
            plt.title(k + "of NeuralNetworks trained on different time Deltas")
            plt.xticks(rotation=90)
            plt.plot(v.keys(), v.values(), color='orange')
            plt.xlabel('Seconds after DHCP broadcast') 
            plt.ylabel(k) 
            plt.savefig(self.plotDir + str(len(self.seeds)) + "_" + k + "_Chart.png")
            plt.clf()


if __name__ == "__main__":
    NC = NetworkCompare(1)
    NC.generateNetworks()
    NC.plot()