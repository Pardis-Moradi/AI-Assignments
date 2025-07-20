# AI Assignments – Sharif University of Technology (2025)

This repository contains a series of practical assignments for the **Artificial Intelligence** course at **Sharif University of Technology**, Spring 2025. Each assignment applies core AI algorithms and methodologies to real-world and game-based problems using Python and Jupyter Notebooks.

---

## 📁 Contents

### 🔹 AI-HW1: **Search Algorithms & Heuristic Optimization**
- **Notebook**: `AI_HW1_Practical.ipynb`
- **Overview**:  
  This assignment consists of two main parts:
  1. **Search Algorithms for Pathfinding**  
     Implements and compares performance of classical search algorithms including:
     - Uninformed: Breadth-First Search (BFS), Depth-First Search (DFS)
     - Informed: Uniform Cost Search (UCS), A*  
     These are evaluated on both weighted and unweighted graphs to find the shortest path.
  2. **Subset Sum Optimization**  
     Explores heuristic search techniques to approximate the optimal subset of numbers summing closest to a target:
     - Brute Force Search  
     - Genetic Algorithms (GA)  
     - Hill Climbing  
     Comparison is made in terms of **accuracy** and **execution time**.

---

### 🔹 AI-HW2: **CSPs, Cryptarithmetic, and Adversarial Search**
- **Notebook**: `AI-HW2-practical.ipynb`
- **Supporting Files**: `map.txt`, `input0.txt`
- **Overview**:  
  This multi-part assignment tackles a diverse set of AI problems:

  1. **Graph Coloring as a CSP – The Kingdom Conflict of Eldoria**  
     Models a geopolitical conflict where rival kingdoms (nodes) must be assigned different ruling factions (colors) while minimizing the total number of factions.  
     Techniques:
     - Constraint Satisfaction Problem formulation  
     - Backtracking Search  
     - Heuristic variable ordering  

  2. **Cryptarithmetic Puzzle Solving**  
     Decodes mathematical puzzles using CSP-based approaches to assign digits to letters under arithmetic constraints.

  3. **Adversarial Search in Othello**  
     Develops an AI agent for playing a simplified version of Othello using:
     - Minimax Algorithm  
     - Alpha-Beta Pruning  
     - Expectimax  
     The goal is to maximize win rate against heuristic or random opponents.

---

### 🔹 AI-HW3: **Probabilistic Reasoning with HMMs and Bayesian Networks**
- **Notebook**: `AI_practical3.ipynb`
- **Dataset**: `dna_dataset.csv`
- **Overview**:  
  Focuses on probabilistic models for sequence and structured data analysis:
  - Implements a **Hidden Markov Model (HMM)** on DNA sequences:
    - Forward Algorithm  
    - Viterbi Algorithm  
  - Constructs and reasons with a **Bayesian Network**:
    - Exact Inference via Enumeration and Variable Elimination  
    - Approximate Inference via Rejection Sampling and Likelihood Weighting  
  These methods are used to analyze and predict biological patterns from DNA data.

---

### 🔹 AI-HW4: **Classification, Regression, and Perceptrons in Vision and NLP**
- **Notebook**: `AI-HW4.ipynb`
- **Supporting Files**: `Images/`, `Sample/`, `regression_plots.png`
- **Overview**:  
  Applies supervised learning algorithms to real-world and synthetic datasets:
  1. **Spam Classification** using Naive Bayes  
     Classifies emails as spam or not based on keyword frequency.
  2. **Customer Purchase Prediction**  
     Uses Decision Trees to predict buying behavior from features such as age and income.
  3. **Regression Models**  
     - Polynomial and Logistic Regression applied to numerical datasets.
     - XOR problem solved using:
       - Single-layer Perceptron (unsuccessful)
       - Multi-layer Perceptron (successful)
  4. **Image Filtering & Edge Detection**  
     Applies custom kernels on images to perform:
     - Grayscale conversion  
     - Gaussian Blurring  
     - Sobel X and Y edge detection

---

### 🔹 AI-HW5: **Reinforcement Learning in Simulated Environments**
- **Notebook**: `HW5 - Practical.ipynb`
- **Supporting Media**: `SARSA.mp4`, `random.mp4`
- **Overview**:  
  Implements model-free reinforcement learning techniques on the `LiffWalking-v1` environment from OpenAI Gymnasium:
  - **SARSA (State–Action–Reward–State–Action)**
  - **Q-Learning**  
  Incorporates multiple exploration strategies:
  - ε-Greedy Policy  
  - Decaying ε Strategy  
  - Greedy Policy  
  Performance is evaluated through convergence rate, policy quality, and animated gameplay comparisons.

---

## 🛠 Requirements
- Python 3.x
- Jupyter Notebook
- Required Libraries:  
  `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `opencv-python`, `seaborn`, `networkx`, `torch`, `imageio`, `gymnasium`
