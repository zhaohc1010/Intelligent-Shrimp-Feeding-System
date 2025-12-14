# 🦐 Intelligent Shrimp Feeding System: A Multimodal AI Approach to Precision Aquaculture 🌊

**Intelligent Shrimp Feeding System** is a pioneering, closed-loop solution for precision aquaculture that integrates advanced Artificial Intelligence (AI) with industrial hardware control. Designed to optimize shrimp farming efficiency, this system bridges the gap between digital decision-making and physical execution.


## 🌟 Project Overview

Traditional shrimp farming often relies on manual observation and fixed feeding schedules, leading to inefficiencies and potential health risks for the livestock. Our system revolutionizes this by combining two powerful subsystems:

1.  **🧠 The Brain (`feeding_agent_webapp`)**: A sophisticated software decision-support system. It leverages a hybrid model combining LightGBM machine learning, biological formulas, and Large Language Models (LLMs) to calculate precise feeding amounts based on real-time data and expert knowledge.
2.  **🦾 The Body (`intelligent_feeding_tray`)**: A robust hardware control interface. It utilizes Computer Vision (CV) and Voice Interaction to physically manage the feeding tray, enabling automated adjustments and visual health inspections of the shrimp.

## 📂 Repository Structure

The project is organized into two primary directories, each dedicated to a specific component of the system:

```text
Intelligent-Shrimp-Feeding-System/
│
├── README.md                   # You are here! General project overview.
│
├── feeding_agent_webapp/       # 🧠 Software: Decision Support System
│   ├── app.py                  # Flask web application entry point
│   ├── core_logic.py           # Core algorithms (LightGBM, Formula, LLM)
│   ├── main_agent.py           # CLI agent for terminal interaction
│   ├── translations.py         # UI internationalization (English/Chinese)
│   ├── config.py               # Configuration settings (API keys, paths)
│   ├── requirements.txt        # Python dependencies for the webapp
│   └── README.md               # Detailed documentation for the software component
│
└── intelligent_feeding_tray/   # 🦾 Hardware: Physical Control Interface
    ├── app.py                  # Main control loop for Vision and Voice
    ├── relay_motor_controller.py # Low-level driver for relay/motor control
    ├── config.py               # Configuration for hardware API keys
    ├── requirements.txt        # Python dependencies for the hardware
    └── README.md               # Detailed documentation for the hardware component
```

-----

## 🧠 Part 1: The Brain - Smart Feeding Decision Agent (`feeding_agent_webapp`)

This component acts as the intelligent core of the system, processing data to make expert-level feeding decisions.

### ✨ Key Software Features

  * **Dual-Core Prediction Engine**: Simultaneously calculates feeding amounts using a pre-trained **LightGBM** machine learning model and standard biological **Formulas**, providing a robust baseline for decision-making.
  * **AI-Powered "Chief Expert"**: Utilizes an **OpenAI/DashScope Large Language Model (LLM)** to analyze the outputs from both prediction models. It incorporates user remarks (e.g., "shrimp molting") and historical trends to refine the final feeding recommendation, mimicking human expert reasoning.
  * **Multilingual Web Interface**: Features a user-friendly Web UI built with Flask, supporting both **English** and **Chinese** for broader accessibility.
  * **Comprehensive Logging**: Supports data persistence via **Firebase Firestore** for cloud-based history management and local **Excel** logging for offline use.

### 🚀 Quick Start (Software)

1.  **Navigate to the directory**:
    ```bash
    cd feeding_agent_webapp
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Web Interface**:
    ```bash
    python app.py
    ```
    Access the dashboard at `http://localhost:5001`.

*For detailed setup instructions, including API configuration and CLI usage, please refer to the [feeding\_agent\_webapp/README.md](https://www.google.com/search?q=feeding_agent_webapp/README.md).*

-----

## 🦾 Part 2: The Body - Intelligent Feeding Tray (`intelligent_feeding_tray`)

This component translates digital decisions into physical actions, providing a tangible interface for farm management.

### ✨ Key Hardware Features

  * **Voice-Activated Control**: Parses natural language commands (e.g., "Lift tray", "Stop motor") using Automatic Speech Recognition (ASR) to control hardware relays, allowing for hands-free operation.
  * **Computer Vision Integration**: Captures real-time video frames for visual analysis using Vision Language Models (VLM). This allows the system to visually inspect the tray and answer queries like "Is the tray empty?".
  * **Industrial Hardware Control**: Includes a custom driver module for **LCUS USB Relays**, enabling precise control over DC motors for tray movement with built-in safety mechanisms to prevent overheating.

### 🚀 Quick Start (Hardware)

1.  **Navigate to the directory**:
    ```bash
    cd intelligent_feeding_tray
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Control System**:
    ```bash
    python app.py
    ```
    Follow the on-screen instructions to interact via voice or keyboard commands.

*For detailed hardware requirements and connection guides, please refer to the [intelligent\_feeding\_tray/README.md](https://www.google.com/search?q=intelligent_feeding_tray/README.md).*

-----

## 🤝 Contributing

We welcome contributions to improve the Intelligent Shrimp Feeding System\! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
