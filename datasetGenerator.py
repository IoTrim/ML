import sqlite3
import pandas as pd
from DatabaseConnection import DatabaseConnection
from PcapReader import PcapReader
from collections import defaultdict
import os

class datasetGenerator:

    def __init__(self, tO):
        self.timeOffset = tO
        self.dbCon = DatabaseConnection()
        self.devices = {}
        self.data = []
        self.deviceData = defaultdict(list)
        self.uniqueDns = set()
        self.dataset = []


    def loadDevicesFromDB(self):
        self.dbCon.DBup().dictionaryCursor(True)

        sqlString = """ SELECT
                            device_type, mac_address
                        FROM
                            device
                        LEFT JOIN
                            device_type
                        ON
                            device.device_type = device_type.name
                        ORDER BY
                            device_type
                        DESC; """

        query = self.dbCon.executeSql(sqlString)

        for d in query:
            typ = d['device_type']
            self.devices[typ] = {"mac":d['mac_address']}

        self.dbCon.DBdown()
        return self

    
    def readPcaps(self):
        for k, v in self.devices.items():
            pcapDir = "/home/oli/project/data/traffic/by-mac/" + v["mac"] + "/ctrl3/"
            files = os.listdir(pcapDir)
            for f in files:
                pR = PcapReader(self.timeOffset, pcapDir + f)
                #pR = PcapReader(self.timeOffset, "/home/oli/project/data/ctrl3/2021-07-10_15.08.43_192.168.5.6.pcap")
                self.deviceData[k].extend(pR.parsePcap().computeData().data)
                self.uniqueDns |= pR.allDnsQueries
        return self

    def createDataset(self):
        for k, v in self.deviceData.items():
            for data in v:
                for url in data:
                    row = [k]
                    for col in sorted(self.uniqueDns):
                        if col in url:
                            row.append(url[col])
                        else:
                            row.append(0)
                            
                    self.dataset.append(row)
        return self


    def createDataframe(self):
        dfCols = ["device"] + list(sorted(self.uniqueDns))
        self.df = pd.DataFrame(self.dataset, columns=dfCols)
        self.df.to_csv(f'datasets/{self.timeOffset}_s_dataset.csv')

if __name__ == "__main__":
    dG = datasetGenerator(4).loadDevicesFromDB().readPcaps().createDataset().createDataframe()