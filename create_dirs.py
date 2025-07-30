import os

for d in ("data", "figures", "logs"):
    os.makedirs(d, exist_ok=True)
