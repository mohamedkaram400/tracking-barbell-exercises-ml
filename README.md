# 📊 Tracking Barbell Exercises

This project processes motion sensor data (accelerometer & gyroscope) to detect and count exercise repetitions using signal processing techniques. It applies **low-pass filtering** and **peak detection** to extract meaningful patterns.

---

## 🚀 Features
- 📌 **Data Preprocessing:** Cleans and structures raw sensor data.
- 🏋️‍♂️ **Exercise Classification:** Identifies different exercises such as bench press, squat, deadlift, barbellraw, overhead press.
- 🔢 **Repetition Counting:** Detects and counts exercise repetitions based on motion patterns.
- 📉 **Visualization:** Plots sensor data before and after filtering.
- ⚙️ **Customizable Parameters:** Modify filtering parameters for different exercises.

- 🔢 **Repetition Counting:** Detects and counts the number of repetitions performed.
- 📊 **Performance Evaluation:** Compares predicted repetitions with actual values.
---

## 🔍 How It Works

1. **Data Collection:** Motion sensors record acceleration and rotation during exercises.
2. **Data Processing:** The data is cleaned and structured into categories.
3. **Filtering:** A **low-pass filter** removes noise, making movements clearer.
4. **Peak Detection:** The code detects the high points in movement (like the top of a squat or bench press).
5. **Counting Repetitions:** The detected peaks represent the number of repetitions.
6. **Visualization:** Plots show how the filtering works and how repetitions are counted.

---

## 🎯 Why This Project?

- 📌 Helps athletes and trainers **track exercise performance automatically**.
- 🔍 Improves accuracy by **removing unwanted noise** from raw sensor data.
- 📉 Provides **visual feedback** on exercise movements.
- 🛠️ Customizable for different exercises and intensity levels.

---


## 🛠 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/mohamedkaram400/tracking-barbell-exercises-ml

cd tracking-barbell-exercises
```

### 2️⃣ Create and Activate a Conda Environment

```bash
conda create --name tracking-barbell-exercises python=3.10 -y
conda tracking-barbell-exercises
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 📌 Customization
You can tweak the filtering parameters
```
count_reps(dataset, column="acc_r", cutoff=0.4, order=10)
```
Modify cutoff and order to optimize detection accuracy.

### 🛠 Troubleshooting
### ❓ Common Issues
```
ModuleNotFoundError: Run pip install -r requirements.txt
Data file not found: Ensure data/interim/01_data_processed.pkl exists.
```

### 📜 License
This project is open-source under the MIT License.

Now you can just **copy and paste** this into your `README.md` file! 🚀 Let me know if you need any modifications. 😊

