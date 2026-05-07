import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("results.csv")

# Plot box_loss train vs validation
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['val/box_loss'], label='Box Loss (val)', color='orange')
plt.plot(df.index, df['train/box_loss'], label='Box Loss (train)', color='blue')

plt.xlabel("Epoch")
plt.ylabel("Box Loss")
plt.title("Box Loss: Training vs Validation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("graph.png")