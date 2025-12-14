
# 🦐 Intelligent Shrimp Feeding System: A Multimodal AI Approach to Precision Aquaculture 🌊

[![Python](https://img.shields.io/badge/python-3.9+-yellow.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📖 Introduction

**Intelligent Shrimp Feeding System** is a pioneering, closed-loop solution for precision aquaculture that integrates advanced Artificial Intelligence (AI) with industrial hardware control. Designed to optimize shrimp farming efficiency, this system bridges the gap between digital decision-making and physical execution.

This repository houses the official source code for the research paper: ****.

### System Architecture
The project is divided into two core subsystems working in tandem:

1.  **🧠 The Brain (`feeding_agent_webapp`)**: A decision-support software that uses **LightGBM**, **Formulas**, and **LLMs** to calculate the optimal feeding amount.
2.  **🦾 The Body (`intelligent_feeding_tray`)**: A hardware control interface that uses **Computer Vision** and **Voice Interaction** to physically manage the feeding tray.

---

## 📂 Directory Structure

```text
Intelligent-Shrimp-Feeding-System/
│
├── feeding_agent_webapp/       # [Software] Decision Support System (Flask/LightGBM/LLM)
│   ├── app.py                  # Web Server Entry Point
│   ├── main_agent.py           # CLI Agent Entry Point
│   ├── core_logic.py           # Core Algorithms
│   ├── config.py               # API & Path Configuration
│   └── ...
│
├── intelligent_feeding_tray/   # [Hardware] Physical Control Interface (Vision/Voice)
│   ├── app.py                  # Hardware Control Loop
│   ├── relay_motor_controller.py # Serial Driver for Relay
│   ├── config.py               # API Configuration
│   └── ...
│
└── requirements.txt            # (Optional) Global dependencies
````

-----

## 🚀 Part 1: The Brain (`feeding_agent_webapp`)

This subsystem acts as the intelligent core. It processes water quality data and shrimp biometrics to make expert-level feeding decisions.

### 📋 Prerequisites

  * **LightGBM Model**: Ensure `best_lightgbm_model_ALLD4.joblib` is in the `feeding_agent_webapp/` directory.
  * **Knowledge Base**: Ensure `Feeding rules.docx` is in the `feeding_agent_webapp/` directory.

### ⚙️ Setup & Configuration

1.  **Install Dependencies**:

    ```bash
    cd feeding_agent_webapp
    pip install -r requirements.txt
    ```

2.  **Configure API Keys (`config.py`)**:
    Open `feeding_agent_webapp/config.py` and strictly follow these steps:

      * **LLM API**: You must set your DashScope (or OpenAI-compatible) API key.
          * *Action*: Set the environment variable `DASHSCOPE_API_KEY` on your system, OR modify `core_logic.py` to accept the key directly (not recommended for public repos).
      * **Paths**: Ensure `BASE_DIR` paths match your local file structure.

3.  **Configure Firebase (Optional for Web App)**:
    If you want to use the cloud history feature in the Web App:

      * *Action*: Place your Firebase credentials JSON content into an environment variable named `FIREBASE_CREDENTIALS_JSON`, or modify `app.py` to load from a local file.

### 🖥️ How to Run

**Option A: Web Interface (Visual Dashboard)**
Recommended for visualization and history management.

```bash
python app.py
```

  * Open your browser at `http://localhost:5001`.
  * **Features**: Input forms, history tables, bilingual support (EN/ZH).

**Option B: Command Line Agent (Quick Test)**
Recommended for quick calculations without a web server.

```bash
python main_agent.py
```

  * Follow the text prompts to input parameters like `average_water_temp`, `weight`, etc.

-----

## 🦾 Part 2: The Body (`intelligent_feeding_tray`)

This subsystem handles the physical execution. It listens to voice commands to control motors and uses a camera to inspect the tray.

### 🛠️ Hardware Requirements

  * **PC/Controller**: Windows (Recommended for COM port ease) or Linux.
  * **USB Relay**: LCUS-1, LCUS-2, or LCUS-4 type (Serial communication).
  * **Camera**: Standard USB Webcam.
  * **Audio**: Microphone and Speaker.

### ⚙️ Setup & Configuration

1.  **Install Dependencies**:

    ```bash
    cd ../intelligent_feeding_tray  # If you are in webapp folder
    pip install -r requirements.txt
    ```

    *Note: You must have [FFmpeg](https://ffmpeg.org/) installed on your system for audio processing.*

2.  **Configure Hardware Port (`app.py`)**:

      * *Action*: Check your Device Manager to find the COM port of your USB Relay (e.g., `COM3`, `/dev/ttyUSB0`).
      * Open `app.py` (or `relay_motor_controller.py`) and update the initialization line:
        ```python
        motor_controller = RelayMotorController(port='COM3') # Change 'COM3' to your actual port
        ```

3.  **Configure AI Services (`config.py`)**:
    Open `intelligent_feeding_tray/config.py` and fill in your API keys:

    ```python
    vision_model_config = {
      "api_key": "YOUR_VOLCENGINE_API_KEY", # Replace this
      "model_endpoint": "YOUR_ENDPOINT_ID"  # Replace this
    }

    asr_model_config = {
      "app_key": "YOUR_APP_KEY",           # Replace this
      "access_key": "YOUR_ACCESS_KEY"      # Replace this
    }
    ```

### 🕹️ How to Run

```bash
python app.py
```

### 🗣️ Voice Commands Guide

Hold the **[Spacebar]** to speak. The system supports fuzzy matching for the following commands (English/Chinese):

| Command Intent | Keywords (English) | Keywords (Chinese) | Action |
| :--- | :--- | :--- | :--- |
| **Move Up** | `up`, `raise`, `lift` | 上升, 升高 | Motor rotates to lift tray ($t$ seconds). |
| **Move Down** | `down`, `lower`, `drop` | 下降, 降低 | Motor rotates to lower tray ($t$ seconds). |
| **Power On** | `power on`, `turn on` | 打开电源 | Engages main power relay. |
| **Visual QA** | *(Any other query)* | *(任意询问)* | Captures photo & answers (e.g., "Is it empty?"). |

-----

## ⚠️ Important Notes for Reviewers/Users

1.  **Simulation Mode**: If you do not have the specific USB Relay hardware connected, `intelligent_feeding_tray` will automatically enter **Simulation Mode**. You will see logs in the console instead of physical motor movements.
2.  **API Keys**: The `config.py` files in this repository contain placeholders. You **must** replace them with valid API keys (Volcengine/DashScope) for the AI features to function.

## 📜 Citation

If you use this code or system in your research, please cite our paper:

```bibtex
@article{YourName2025SmartFeeding,
  title={Intelligent Shrimp Feeding System: A Multimodal AI Approach to Precision Aquaculture},
  author={Your Name and Co-authors},
  journal={Journal Name / Conference},
  year={2025}
}
```



```
```
