import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load & Prep Data
df = pd.read_csv("train_data.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale Data (Normalizes big numbers like ports)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save Scaler for later use!
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Convert to Tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# 2. Define Network
class SentryNet(nn.Module):
    def __init__(self):
        super(SentryNet, self).__init__()
        self.fc1 = nn.Linear(4, 32)  # 4 Inputs -> 32 Hidden
        self.fc2 = nn.Linear(32, 16) # 32 -> 16 Hidden
        self.fc3 = nn.Linear(16, 2)  # 16 -> 2 Output (Safe/Bad)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SentryNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 3. Train
print("Training Neural Network...")
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

# 4. Save Model
torch.save(model.state_dict(), "sentry_brain.pth")
print("Model saved as sentry_brain.pth")
