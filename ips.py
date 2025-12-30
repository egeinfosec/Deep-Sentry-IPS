from scapy.all import *
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import sys

# --- CONFIG ---
INTERFACE = "wlan0"
CONFIDENCE_THRESHOLD = 0.8  # Only block if 80% sure

# --- LOAD RESOURCES ---
class SentryNet(nn.Module):
    def __init__(self):
        super(SentryNet, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
    def forward(self, x): return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

print("[*] Loading AI Model...")
model = SentryNet()
model.load_state_dict(torch.load("sentry_brain.pth"))
model.eval() # Set to evaluation mode
scaler = pickle.load(open("scaler.pkl", "rb"))

blocked_ips = set()

def block_ip(ip):
    if ip not in blocked_ips:
        print(f"\n[!!!] BLOCKING MALICIOUS IP: {ip}")
        os.system(f"iptables -A INPUT -s {ip} -j DROP")
        blocked_ips.add(ip)

def process_packet(pkt):
    if IP in pkt and TCP in pkt: # Focus on TCP for now
        src_ip = pkt[IP].src
        
        # Whitelist yourself!
        if src_ip == "192.168.1.77": return 

        if src_ip in blocked_ips: return

        # Extract features (Must match preprocess.py)
        features = [len(pkt), pkt[IP].proto, pkt[TCP].dport, int(pkt[TCP].flags)]
        
        # Scale & Convert
        features_scaled = scaler.transform([features])
        tensor_in = torch.FloatTensor(features_scaled)

        # Predict
        with torch.no_grad():
            outputs = model(tensor_in)
            probs = torch.softmax(outputs, dim=1)
            attack_prob = probs[0][1].item()

        if attack_prob > CONFIDENCE_THRESHOLD:
            print(f"[!] Detection: {src_ip} (Confidence: {attack_prob:.2f})")
            block_ip(src_ip)

print(f"[*] Deep-Sentry IPS Running on {INTERFACE}...")
sniff(iface=INTERFACE, prn=process_packet, store=0)
