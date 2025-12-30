# 🛡️ Deep-Sentry: AI-Powered Active IPS

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-green)
![Status](https://img.shields.io/badge/Status-Prototype-orange)

## 📖 Overview
**Deep-Sentry** is an active Intrusion Prevention System (IPS) that runs on a Raspberry Pi 5. Unlike traditional firewalls that use static rules, Deep-Sentry uses a **Neural Network** to inspect network packets in real-time. If it detects malicious behavior (like Nmap scans), it automatically updates the Linux Kernel Firewall (`iptables`) to block the attacker.

## ⚙️ Architecture
1.  **Packet Sniffing:** Uses `Scapy` to capture raw traffic on `wlan0`.
2.  **Feature Extraction:** Converts packets into vectors (Size, Protocol, Flags, Port).
3.  **AI Inference:** A PyTorch Deep Learning model analyzes the traffic.
4.  **Active Defense:** If Confidence > 80%, the source IP is banned via `iptables`.

## 🚀 Installation & Setup

### 1. Requirements
* Raspberry Pi 5 (running Kali Linux)
* Type-C Power Cable
* SSH Connection (Laptop & Pi must be on the same network)

### 2. Environment Setup
```bash
sudo apt update && sudo apt install -y python3-venv tcpdump
mkdir ~/deep-sentry && cd ~/deep-sentry
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision scapy pandas numpy scikit-learn
```
## 🧠 Training the Brain
The model was trained on live traffic captured from the device:

Benign Data: Captured normal background web traffic (normal.pcap).

Malicious Data: Captured nmap -sS and nmap -A scans (attack.pcap).

Training: The model achieved high accuracy in distinguishing normal packets from scan probes.

## 🛡️ Usage
To start the Intrusion Prevention System:

```bash
sudo ./venv/bin/python ips.py
```
Verification
When an attack is launched from a non-whitelisted IP (e.g., 192.168.1.66), the system detects and blocks it immediately:

```Plaintext

[!] Detection: 192.168.1.66 (Confidence: 1.00)
[!!!] BLOCKING MALICIOUS IP: 192.168.1.66
```
⚠️ Note on Connectivity: Since the attack was launched from the same machine used for SSH (192.168.1.66), the IPS correctly identified the threat and severed the connection immediately. This confirms the firewall rule was applied in real-time.

## 👨‍💻 Author
egeinfosec
