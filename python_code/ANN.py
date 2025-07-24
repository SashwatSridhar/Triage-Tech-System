#importing and definnig libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

# Use these for your model
Sequential = tf.keras.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
to_categorical = tf.keras.utils.to_categorical

import matplotlib.pyplot as plt

#loading the dataset from the excel file
excel_file = "sensor_data_training.xlsx"
df = pd.read_excel(excel_file)

print(f"Excel file path: {excel_file}")
print(f"Initial dataframe shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")
print(f"First few rows of data:")
print(df.head())

# Check for missing values
print(f"Number of missing values per column:\n{df.isna().sum()}")

# Fill missing Age values with the median age or a default value if all are missing
if pd.isna(df['Age'].median()):
    print("All Age values are missing, filling with default value of 50")
    df['Age'].fillna(50, inplace=True)
else:
    df['Age'].fillna(df['Age'].median(), inplace=True)

print(f"Shape after filling missing values: {df.shape}")

# Feature selection (independent variables)
required_columns = ["Temperature (°C)", "SpO₂ (%)", "Blood Pressure (mmHg)", "Heart Rate (BPM)", "Age"]
X = df[required_columns].values
print(f"X shape after feature selection: {X.shape}")
print(f"First few rows of X:")
print(X[:5])

# Define labels (classification based on predefined thresholds)
def classify_patient(temp, spo2, bp, hr, age):
    if (temp >= 39 or temp <= 35) or spo2 < 90 or bp > 180 or hr > 100 or age > 65:  # Critical condition
        return 0  # Class A (Emergency)
    elif ((38 <= temp < 39) or (35 < temp <= 37)) or (90 <= spo2 <= 94) or (140 <= bp <= 180) or (85 <= hr <= 99) or (50 <= age <= 65):  # Moderate
        return 1  # Class B (Urgent)
    else:  # Normal readings
        return 2  # Class C (Non-Urgent)
    
# Apply classification function
y = np.array([classify_patient(row[0], row[1], row[2], row[3], row[4]) for row in X])
print(f"y shape: {y.shape}")
print(f"Unique classes in y: {np.unique(y)}")
print(f"Class distribution: {np.bincount(y)}")

y = to_categorical(y, num_classes=3)
print(f"y shape after to_categorical: {y.shape}")

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data processing complete")

# Build MLP model(Multi-layer Perceptron)
model = Sequential([
    Dense(32, activation='relu', input_shape=(5,)),  # First hidden layer
    Dropout(0.2),  # Dropout for regularization
    Dense(16, activation='relu'),  # Second hidden layer
    Dropout(0.2),
    Dense(3, activation='softmax')  # Output layer (3 classes)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

print("Model training complete!")

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot training & validation 
plt.figure(1, figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Accuracy Over Epochs")
plt.show()

# Plot training & validation loss
plt.figure(2, figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Model Loss Over Epochs")
plt.show()

#------------patient prediction-----------#
patient_file = "sensor_data.xlsx"
patient_df = pd.read_excel(patient_file)

# Check for missing values in patient data
print(f"\nPatient data - missing values per column:\n{patient_df.isna().sum()}")

# Fill missing Age values in patient data
if 'Age' in patient_df.columns and patient_df['Age'].isna().any():
    patient_df['Age'].fillna(50, inplace=True)
    print("Filled missing Age values in patient data")

# Remove rows with invalid or zero values
patient_df = patient_df[(patient_df != 0).all(axis=1)]
print(f"Patient data shape after removing zeros: {patient_df.shape}")

# Prediction function
def predict_patient_severity(temp, spo2, bp, hr, age):
    new_data = np.array([[temp, spo2, bp, hr, age]])
    new_data = scaler.transform(new_data)
    prediction = model.predict(new_data)
    predicted_class = np.argmax(prediction)

    classes = {
        0: ("Emergency (Class A)", 1, "0-5 minutes"),
        1: ("Urgent (Class B)", 2, "10-20 minutes"),
        2: ("Non-Urgent (Class C)", 3, "30+ minutes")
    }
    return classes[predicted_class]

# Check if column names match between training and prediction
print(f"\nPatient data columns: {patient_df.columns.tolist()}")
bp_column = 'Blood Pressure (mmHg)' if 'Blood Pressure (mmHg)' in patient_df.columns else 'Blood Pressure(mmHg)'

# Apply prediction to each patient
severities = []
priorities = []
wait_times = []

for index, row in patient_df.iterrows():
    temp = row['Temperature (°C)']
    spo2 = row['SpO₂ (%)']
    bp = row[bp_column]
    hr = row['Heart Rate (BPM)']
    age = row['Age']

    severity, priority, wait_time = predict_patient_severity(temp, spo2, bp, hr, age)
    
    severities.append(severity)
    priorities.append(priority)
    wait_times.append(wait_time)

# Add results to the DataFrame
patient_df['Severity'] = severities
patient_df['Priority'] = priorities
patient_df['Estimated Wait Time'] = wait_times

# Sort patients by priority (1 = highest)
priority_list = patient_df.sort_values(by='Priority')

# Display the priority list
print("\n PRIORITY LIST:")
print(priority_list[['Temperature (°C)', 'SpO₂ (%)', bp_column, 'Heart Rate (BPM)', 'Age', 'Severity', 'Priority', 'Estimated Wait Time']])

# Save the priority list back to Excel
output_file = "patient_priority_list.xlsx"
priority_list.to_excel(output_file, index=False)
print(f"\n Priority list saved to '{output_file}'")