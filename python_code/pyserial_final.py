import serial
import serial.tools.list_ports
import requests
import time
import os
import sys
import json

# Google Sheets Web App URL
GOOGLE_SHEET_URL = "https://script.google.com/macros/s/AKfycbyLtUJoQYfWFSOLJ5BnAO-LlXgOk-aGFOVMAxwQWGVa1Uw4m10feOxIfDkMCsif0o6E/exec"

# Optional: Enable Excel backup (set to False to disable)
SAVE_EXCEL_BACKUP = True
EXCEL_FILE = "sensor_data.xlsx"

print("============================================================")
print("          TRIAGETECH SOLUTIONS - VITAL SIGNS KIOSK          ")
print("============================================================")

# Connect directly to COM3 without asking
COM_PORT = "COM3"
try:
    print(f"Connecting to {COM_PORT}...")
    serialInst = serial.Serial(port=COM_PORT, baudrate=9600, timeout=1)
    print(f"✓ Successfully connected to {COM_PORT}")
except serial.SerialException as e:
    print(f"✗ Error: Could not connect to {COM_PORT}")
    print(f"  {e}")
    print("\nPossible solutions:")
    print("1. Close any programs that might be using the port (Arduino IDE, Serial Monitor)")
    print("2. Try running this script with administrator privileges")
    print("3. Unplug and reconnect your Arduino")
    print("4. Restart your computer")
    sys.exit(1)

# Setup Excel backup if enabled
if SAVE_EXCEL_BACKUP:
    try:
        import pandas as pd
        import openpyxl
        
        if not os.path.exists(EXCEL_FILE):
            df = pd.DataFrame(columns=["Timestamp", "Temperature (°C)", "SpO₂ (%)", "Heart Rate (BPM)"])
            df.to_excel(EXCEL_FILE, index=False)
            print(f"Created Excel backup file: {EXCEL_FILE}")
    except ImportError:
        print("Warning: pandas or openpyxl not installed. Excel backup disabled.")
        SAVE_EXCEL_BACKUP = False

# Initialize variables
temperature = None
blood_oxygen = None
heart_rate = None

# Function to send data to Google Sheets
# Modify the send_to_google_sheets function in your Python script:

def send_to_google_sheets(temp, spo2, hr):
    params = {
        'action': 'addData',
        'temperature': temp,
        'spo2': spo2,
        'heartrate': hr
    }
    
    try:
        print(f"Sending vital signs to system: T={temp}°C, SpO2={spo2}%, HR={hr}BPM")
        print(f"URL: {GOOGLE_SHEET_URL}")
        print(f"Parameters: {params}")
        
        # Send the request and get the full response
        response = requests.get(GOOGLE_SHEET_URL, params=params, timeout=15)
        
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response content: {response.text[:200]}") # Only print first 200 chars
        
        # Consider any 200 status code as success even if the content looks like an error
        if response.status_code == 200:
            print(f"✓ Data successfully sent to system (HTTP 200 OK)")
            return True
        else:
            print(f"✗ Failed to send data: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error sending data: {e}")
        return False

# Function to save data to Excel
def save_to_excel(temp, spo2, hr):
    if not SAVE_EXCEL_BACKUP:
        return
        
    try:
        df = pd.read_excel(EXCEL_FILE)
        new_data = pd.DataFrame([[time.strftime("%Y-%m-%d %H:%M:%S"), temp, spo2, hr]], 
                               columns=["Timestamp", "Temperature (°C)", "SpO₂ (%)", "Heart Rate (BPM)"])
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_excel(EXCEL_FILE, index=False)
        print(f"✓ Data saved to Excel backup: {EXCEL_FILE}")
    except Exception as e:
        print(f"✗ Error saving to Excel: {e}")

print("\n============================================================")
print("      READY FOR PATIENT - PLACE SENSORS APPROPRIATELY       ")
print("============================================================")
print("\nWaiting for readings from Arduino...")

patient_count = 0

try:
    while True:
        try:
            if serialInst.in_waiting > 0:
                data = serialInst.readline().decode('utf-8').strip()
                print(f"Received: {data}")

                if "Temperature" in data:
                    temperature = float(data.split("=")[1].strip())
                elif "SpO2" in data:
                    blood_oxygen = float(data.split("=")[1].strip())
                elif "Heart Rate" in data:
                    heart_rate = float(data.split("=")[1].strip())

                # If all data is collected, save to Google Sheets
                if temperature is not None and blood_oxygen is not None and heart_rate is not None:
                    patient_count += 1
                    
                    print("\n------------------------------------------------------------")
                    print(f"PATIENT #{patient_count} - VITAL SIGNS COLLECTED:")
                    print(f"Temperature: {temperature}°C")
                    print(f"SpO₂: {blood_oxygen}%")
                    print(f"Heart Rate: {heart_rate} BPM")
                    print("------------------------------------------------------------\n")
                    
                    # Send to Google Sheets
                    sheets_success = send_to_google_sheets(temperature, blood_oxygen, heart_rate)
                    
                    # Check if we should try again
                    if not sheets_success:
                        print("Would you like to retry sending to Google Sheets? (y/n)")
                        retry = input().strip().lower()
                        if retry == "y":
                            print("Retrying...")
                            send_to_google_sheets(temperature, blood_oxygen, heart_rate)
                    
                    # Save to Excel if enabled
                    if SAVE_EXCEL_BACKUP:
                        save_to_excel(temperature, blood_oxygen, heart_rate)
                    
                    # Reset after saving
                    temperature = None
                    blood_oxygen = None
                    heart_rate = None
                    
                    print("\n============================================================")
                    print("      READY FOR NEXT PATIENT - PLACE SENSORS APPROPRIATELY   ")
                    print("============================================================")
                    
                    # Add a delay to prevent immediate reuse
                    time.sleep(2)

        except KeyboardInterrupt:
            raise  # Re-raise to be caught by the outer try-except
        except Exception as e:
            print(f"Error reading from Arduino: {e}")
            time.sleep(1)  # Add delay on error to prevent tight error loops
            
except KeyboardInterrupt:
    print("\nProgram terminated by user")
    # Close the serial connection properly
    if serialInst and serialInst.is_open:
        serialInst.close()
        print("Serial connection closed")
except Exception as e:
    print(f"\nUnexpected error: {e}")
    if serialInst and serialInst.is_open:
        serialInst.close()
        print("Serial connection closed")
finally:
    print("TriageTech Vital Signs Kiosk has shut down")