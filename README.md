<div align="center">

# 🤖 AI-Based Autonomous Navigation System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge\&logo=python)
![Pygame](https://img.shields.io/badge/Pygame-2.5.2-green?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-red?style=for-the-badge\&logo=opencv)

![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

### 🚗 Autonomous Navigation using AI Concepts, Path Planning & Simulation

A complete simulation of an autonomous system implementing perception, path planning, obstacle avoidance, and real-time navigation — fully runnable on a laptop without GPU.

</div>

---


## 💡 Why This Project

Most academic projects focus only on algorithms.

This project goes beyond theory by building a **complete autonomous navigation pipeline**:

* perception
* path planning
* decision making
* navigation

It demonstrates practical system design similar to real-world robotics and autonomous vehicles.

---

## 📌 Project Overview

This project simulates the core working of an autonomous agent navigating in an environment:

Perceive → Plan → Navigate → Avoid Obstacles → Reach Goal

The system includes:

* 2D grid-based environment
* Virtual autonomous agent
* A* and Dijkstra path planning algorithms
* Sensor-based obstacle detection
* Real-time dynamic replanning
* Interactive simulation using Pygame

This is a fully virtual system — no hardware required.

---

## 🚀 Key Features

* Real-time A* path planning
* Dynamic obstacle avoidance
* Sensor-based perception simulation
* Interactive Pygame environment
* Multiple execution modes (GUI + CLI + Testing)
* Modular and scalable architecture

---

## 🏭 Industry Relevance

This system reflects real-world applications in:

* Autonomous Vehicles (Tesla, Waymo)
* Warehouse Robotics (Amazon Kiva)
* Drone Navigation
* Delivery Robots
* Healthcare Robotics
* Smart Mobility Systems

---

## 🛠️ Tech Stack

| Component       | Technology   |
| --------------- | ------------ |
| Language        | Python 3.10+ |
| Simulation      | Pygame       |
| Algorithms      | A*, Dijkstra |
| Vision Support  | OpenCV       |
| Data Processing | NumPy        |
| Visualization   | Matplotlib   |
| Testing         | pytest       |

---

## 🏗️ Architecture

Grid Environment
↓
Perception (Obstacle Detection)
↓
Path Planning (A*)
↓
Navigation Controller
↓
Agent Movement
↓
Visualization (Pygame / Matplotlib)

---

## 📁 Project Structure

AI-Autonomous-Navigation-System/
│
├── src/
│   ├── simulation/
│   ├── perception/
│   ├── path_planning/
│   ├── navigation/
│   └── utils/
│
├── tests/
├── notebooks/
├── images/
├── videos/
├── outputs/
│
├── main.py
├── requirements.txt
└── README.md

---

## ⚡ Installation

Clone the repository:

git clone https://github.com/Saru2248/AI-Autonomous-Navigation-System.git
cd AI-Autonomous-Navigation-System

Create virtual environment:

python -m venv venv
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

---

## ▶️ Run the Project

Simulation Mode:

python main.py --mode sim

Demo Mode:

python main.py --mode demo --save-output

Run Tests:

pytest tests/ -v

---

## 🎮 Controls (Simulation)

SPACE → Start navigation
Left Click → Add/remove obstacle
R → Random map
C → Clear map
TAB → Switch algorithm
F2 → Set Start
F3 → Set Goal
S → Save screenshot


## 📊 Sample Output

Path Length: 32
Nodes Explored (A*): 110
Nodes Explored (Dijkstra): 280

A* explores fewer nodes, making it more efficient.

---

## 🔬 Simulation Workflow

1. Launch simulation
2. Generate environment
3. Plan path using A*
4. Agent starts moving
5. Sensor detects obstacles
6. Path replanning triggered
7. Agent reaches goal

---

## 🔮 Future Improvements

* YOLO-based object detection
* ROS2 integration
* CARLA simulator (3D environment)
* Reinforcement learning agent
* SLAM (mapping + localization)
* Multi-agent navigation

---

## 📚 Learning Outcomes

* Path planning algorithms (A*)
* Autonomous system pipeline design
* Simulation-based robotics
* Modular software architecture
* Debugging and testing

---

## 🐛 Troubleshooting

pygame not found → pip install pygame
No path found → reduce obstacle density
Black screen → check Python version
pytest not found → pip install pytest

---

## 👨‍💻 Author

Sarthak Dhumal

Email: [svd8007@gmail.com](mailto:svd8007@gmail.com)
GitHub: https://github.com/Saru2248
LinkedIn: https://www.linkedin.com/in/sarthak-dhumal-07555a211/

---

<div align="center">


## 🎬 Demo
https://drive.google.com/file/d/1yp1uVE3cC6GeGkGroWvSJa7-YB6m2AIQ/view?usp=sharing


⭐ Star this repository if you found it useful!


</div>
