"""
    Package imports:

    pandas: Used for reading the CSV file generated during the training of the YOLO model, which contains the training and validation box loss values for each epoch.
    matplotlib.pyplot: Used for plotting the training and validation box loss over epochs to visualize the model's performance during training.
"""
import pandas as pd
import matplotlib.pyplot as plt


"""
    Read the CSV file generated during the training of the YOLO model.
"""
df = pd.read_csv("./runs/detect/y26_1/results.csv")

"""
    Plot the training and validation box loss over epochs to visualize the model's performance during training.
"""
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