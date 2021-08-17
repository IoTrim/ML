from datetime import datetime, date, timedelta
from re import findall
from collections import defaultdict
from scapy.all import *
import numpy as np
import pickle, time, sys

class PcapReader:
    def __init__(self, tO, pcap):
        self.timeOffset = int(tO)
        self.pcapPath = pcap
        self.picklePath = pcap + str(tO) + '.pickle'
        self.ip = re.findall(r'(?:\d{1,3}\.)+(?:\d{1,3})', pcap)[-1]
        self.dhcpBroadcastTimes = set()
        self.dnsQueries = defaultdict(list)
        self.allDnsQueries = set()
        self.data = None
        self.pcap = rdpcap(self.pcapPath)
        self.avgDhcpFreq = -1
        self.debug = False


    def getTime(self, *args, **kwargs):
        return datetime.fromtimestamp(*args, **kwargs)

    def save(self):
        f = open(self.picklePath, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self):
        print("loading pickled object")
        f = open(self.picklePath, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          
        self.__dict__.update(tmp_dict) 

    def parsePcap(self):
        # if os.path.isfile(self.picklePath):
        #     return self
            
        dhcpSrc, broadcast = '0.0.0.0', '255.255.255.255'

        for pkt in self.pcap:
            try:
                if UDP in pkt and pkt[IP].src == dhcpSrc and pkt[IP].dst == broadcast:
                    self.dhcpBroadcastTimes.add(self.getTime(pkt.time))
                elif pkt.haslayer(DNSQR) and pkt[IP].src == self.ip:
                    query = pkt[DNSQR].qname
                    self.dnsQueries[self.getTime(pkt.time)].append(query.decode()[:-1])
            except IndexError:
                pass
                
        self.dhcpBroadcastTimes = list(sorted(self.dhcpBroadcastTimes)) + [datetime.max]

        # compute difference between adjacent items excluding last one
        staggered = zip(self.dhcpBroadcastTimes[:-2], self.dhcpBroadcastTimes[1:-1])
        dhcpDelta = [j.timestamp() - i.timestamp() for i, j in staggered]

        if dhcpDelta:
            self.avgDhcpFreq = sum(dhcpDelta) / len(dhcpDelta)

        if self.debug:
            print("DHCP discover timestamps:", [t.__str__() for t in sorted(self.dhcpBroadcastTimes)])
            print("average time between DHCP discover packets", self.avgDhcpFreq, 's')

        print("PCAP file '", self.pcapPath, "' parsed.")

        return self


    def computeData(self):
        # if os.path.isfile(self.picklePath):
        #     self.load()
        #     return self

        def _countFreq(urls):
            countMap = defaultdict(int)
            for i in urls:
                countMap[i] += 1
            return [{k:v/self.timeOffset} for k, v in countMap.items()]

        self.data = [[] for _ in range(len(self.dhcpBroadcastTimes))]
        for i, time in enumerate(self.dhcpBroadcastTimes[:-1]):
            for sec in range(self.timeOffset):
                t = time + timedelta(seconds = int(sec))
                if t < self.dhcpBroadcastTimes[i + 1]:
                    self.data[i].extend(self.dnsQueries[t])
                    self.allDnsQueries |= set(self.dnsQueries[t])

        self.data = filter(lambda x: x, self.data)
        self.data = map(_countFreq, self.data)

        if self.debug:
            print(self.allDnsQueries)
            print(len(self.allDnsQueries))
            for i in self.data:
                print(i)

        # self.save()

        return self


if __name__ == "__main__":
    start = time.time()
    try:
        pR = PcapReader(*sys.argv[1:])
    except ValueError and TypeError:
        print("Usage is timeDelta, pcapFileName")
        exit(-1)

    pR.debug = True
    data = pR.parsePcap().computeData().data
    end = time.time()
    print ("Time elapsed:", end - start)