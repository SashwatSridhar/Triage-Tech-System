# 🏥 Triage Tech Solutions – AI-Powered Emergency Room Triage Chair

An automated medical triage system that integrates biomedical sensors and machine learning to reduce emergency room wait times and improve patient prioritization.

---

## 📘 Project Summary

Emergency rooms are often overwhelmed, leading to delayed treatments and poorer patient outcomes. This project addresses those challenges by automating the triage process using real-time vital monitoring and AI-based prioritization.

We designed a **smart triage chair** embedded with biomedical sensors and an **Arduino Uno R3**. The system collects a patient’s **heart rate**, **SpO₂**, **body temperature**, and **age** data via a custom user interface. A trained **Artificial Neural Network (ANN)** then classifies patients into urgency categories and estimates wait times—streamlining the triage process for clinical staff.

---

## ⚙️ System Features

- ✅ Touch-free, non-invasive data collection via sensors
- ✅ 30-second vital scan per patient
- ✅ AI classification into 3 priority levels:
  - **Class A** – Emergency
  - **Class B** – Urgent
  - **Class C** – Non-Urgent
- ✅ Estimated wait time output
- ✅ Excel database export for hospital record-keeping
- ✅ Printed queue list for medical staff

---

## 🧠 Machine Learning Pipeline

- **Libraries Used**: TensorFlow, Keras, NumPy, Pandas
- **Model Type**: Artificial Neural Network (ANN)
- **Training**: 80-20 data split over 50 epochs
- **Accuracy**: **96.75%** on test data
- **Inputs**: Heart rate, SpO₂, body temperature, age, sex
- **Outputs**: Triage class + estimated wait time

---

## 🛠️ Technologies

- Arduino Uno R3
- DHT11 Temperature Sensor
- MAX30102 Heart Rate & SpO₂ Sensor
- Python 3.11 (data processing, classification, reporting)
- Serial communication (USB interface)
- Excel (data output with `openpyxl`)

---

## 💻 How It Works

```plaintext
Patient places hand → Sensors collect vitals → Arduino sends data to Python → ANN classifies priority → Wait time is calculated → Queue list is printed and saved
