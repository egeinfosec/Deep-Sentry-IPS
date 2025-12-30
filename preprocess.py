from scapy.all import *
import pandas as pd
import numpy as np

def parse_pcap(file_path, label):
    print(f"Parsing {file_path}...")
    packets = rdpcap(file_path)
    data = []
    
    for pkt in packets:
        # Filter: Only look at IP traffic
        if IP in pkt:
            # 1. Packet Size
            size = len(pkt)
            
            # 2. Protocol (TCP=6, UDP=17)
            proto = pkt[IP].proto
            
            # 3. Destination Port (Attacks often probe random ports)
            dport = 0
            if TCP in pkt: dport = pkt[TCP].dport
            elif UDP in pkt: dport = pkt[UDP].dport
            
            # 4. TCP Flags (SYN=2, ACK=16, etc. Crucial for scan detection)
            flags = 0
            if TCP in pkt: flags = int(pkt[TCP].flags)
            
            data.append([size, proto, dport, flags, label])

    return data

# 0 = Benign, 1 = Malicious
benign = parse_pcap("normal.pcap", 0)
malicious = parse_pcap("attack.pcap", 1)

columns = ["size", "proto", "dport", "flags", "label"]
df = pd.DataFrame(benign + malicious, columns=columns)
df.to_csv("train_data.csv", index=False)
print(f"Dataset created with {len(df)} packets.")
