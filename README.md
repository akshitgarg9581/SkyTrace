---
title: SkyTrace
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SkyTrace: AI-Powered Flight Route Contrail Optimizer

A lightweight, real-time web application and deep learning pipeline designed to detect high-risk contrail-forming zones in the upper atmosphere. By scanning geostationary satellite scans and evaluating planned flight vectors (A ↔ B) against contrail probability grids, SkyTrace helps flight dispatchers plan optimal flight paths with minimal environmental impact.

---

## 🔬 Scientific Context
Contrails (condensation trails) are ice-crystal clouds formed by the water vapor and soot emitted from aircraft engines at high altitudes. While appearing harmless, persistent contrails trap outbound thermal radiation in the atmosphere, accounting for **over 50% of aviation's total warming contribution**—outweighing the impact of cumulative engine CO2 emissions. 

Scientific research shows that minor altitude adjustments (±2,000 feet) or lateral routing changes can prevent contrails completely. SkyTrace provides dispatchers with the forecasting tool needed to execute these path corrections.

---

## 🛠️ The Tech Stack
*   **Deep Learning (PyTorch)**: U-Net architecture with an `efficientnet-b3` pretrained backbone and concurrent Squeeze-and-Excitation (`scse`) decoder attention.
*   **Backend Engine (Flask)**: Performs tensor preprocessing, executes PyTorch model inference, and processes output matrices into Base64-encoded visual heatmaps in under 1 second.
*   **Database & Auth (SQLite)**: SQLite database managing user registry with encrypted credentials (one-way password hashing via Werkzeug).
*   **Frontend Dashboard**: Obsidian dark-space UI featuring interactive HTML5 Canvas drawing, real-time pixel-level risk sampling, and dynamic Chart.js training performance graphs.

---

## 🤖 Machine Learning Pipeline (`model_1.py`)

### 1. Ash-RGB Channel Engineering
Contrails are thin and nearly invisible in standard optical ranges. The model processes multispectral images from the GOES-16 satellite using **Ash-RGB** differential band combinations:
*   **Red Channel**: Band 15 - Band 14 (Measures cloud optical depth)
*   **Green Channel**: Band 14 - Band 11 (Isolates ice crystal size)
*   **Blue Channel**: Band 14 (Calibrates atmospheric temperature)

### 2. Multi-Frame Temporal Stacking (9-Channels)
Static images often confuse thin cirrus clouds with contrails. To give the model temporal context, the data pipeline stacks three sequential timestamps ($t-10\text{min}$, $t$, $t+10\text{min}$), resulting in a **9-channel input tensor**. This allows the model to capture movement and growth vectors.

### 3. Model Training & Validation
*   **Dataset Source**: Official Google Research [Identify Contrails to Reduce Global Warming](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming) Kaggle dataset.
*   **Dataset Volume**: Trained on **8,000 records** and validated on **1,000 records**.
*   **Loss Formulation**: Combined Focal Loss and Dice Loss ($0.5 \times \text{Dice} + 0.5 \times \text{Focal}$) to combat the severe class imbalance of contrail pixels ($<1\%$ of typical frames).
*   **Training Schedule**: 25 epochs optimized using the `OneCycleLR` learning rate schedule.
*   **Validation Metrics**: Peak Validation **Dice Score: 0.4827**.

---

## ⚡ Latency-Free Route Risk Sampling
Rather than sending every coordinate change back to the server as a dispatcher draws a flight path on the screen, SkyTrace uses a client-side optimization:
1.  On image upload, the Flask backend returns both the visual prediction overlay and a raw grayscale probability map (`probmap`).
2.  The frontend loads this probability map into a hidden off-screen HTML5 canvas.
3.  As the user draws a vector (A ↔ B) on the visible canvas, JavaScript samples 100 points along that vector directly from the off-screen canvas's image data buffer using `getImageData()`.
4.  This client-side calculation determines average risk, peak probability, and hotspot intersections in under **10ms**, rendering real-time danger warnings (`🚨 HIGH RISK`) or optimal flight path clearances instantly.

---

## 💻 Quick Start & Installation

### 1. Environment Setup
Install dependencies:
```bash
pip install -r requirements_web.txt
```

### 2. Run the Application Locally
Run the Flask server:
```bash
python app.py
```
*Note: The application will automatically create a local `users.db` database on its first launch.*

By default, the server binds to port `7860`. Open your browser and navigate to:
```
http://localhost:7860
```
Register an account, upload a satellite scan, and draw flight paths to check predictions.

### 🐳 Containerized Deployment (Docker)
Build the image:
```bash
docker build -t skytrace .
```
Run the container:
```bash
docker run -p 7860:7860 skytrace
```

---

## 📂 Repository Layout

*   `app.py`: Flask application routes, authentication gates, and base64 matplotlib overlay engine.
*   `database.py`: SQL database configuration and path security decorators.
*   `model_1.py`: PyTorch dataset classes, augmentations, and U-Net training configurations.
*   `contrail_model.pth`: Trained model weight checkpoint file.
*   `history.json`: Validation and training history metrics over 25 epochs.
*   `Dockerfile`: Build recipe for Docker and Hugging Face container deployments.
*   `templates/`: User login, signup, and dashboard templates.
*   `static/`: Dashboard styling (`style.css`) and client-side canvas route calculation script (`main.js`).
