import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utlis import *
import tensorflow as tf


# Step 1: Initialize data
print('Initializing data')
data_path = 'DataCollected'
data_info = importDataInfo(data_path)
print(data_info.head())

# Step 2: Visualize and balance data
print('Visualizing and balancing data')
balanced_data = balanceData(data_info, display=True)

# Step 3: Prepare for processing
image_paths, steerings = loadData(data_path, balanced_data)

# Step 4: Split for training and validation
x_train, x_val, y_train, y_val = train_test_split(image_paths, steerings,
                                                  test_size=0.2, random_state=10)
print(f'Total Training Images: {len(x_train)}')
print(f'Total Validation Images: {len(x_val)}')

# Step 5: Augment data

# Step 6: Preprocess

# Step 7: Create model
print('Creating model')
model = createModel()

# Step 8: Training
print('Training model')
history = model.fit(dataGen(x_train, y_train, batch_size=100, training=1),
                    steps_per_epoch=100,
                    epochs=10,
                    validation_data=dataGen(x_val, y_val, batch_size=50, training=0),
                    validation_steps=50)

# Step 9: Save the model
model.save('model.h5')
print('Model saved')

# Step 10: Plot the results
def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.show()

plot_history(history)
