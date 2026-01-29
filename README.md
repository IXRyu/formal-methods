# Process Mining Analysis - BPI Challenge 2012

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PM4Py](https://img.shields.io/badge/PM4Py-Process%20Mining-green.svg)](https://pm4py.fit.fraunhofer.de/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-purple.svg)](https://openai.com/)

A comprehensive Process Mining project analyzing the **BPI Challenge 2012** dataset (Financial Loan Application Process) using multiple discovery algorithms, conformance checking, and LLM-powered analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Files](#output-files)
- [Algorithms Used](#algorithms-used)
- [License](#license)

## 🎯 Overview

This project applies Process Mining techniques to the BPI Challenge 2012 dataset, which contains real event logs from a personal loan application process at a Dutch financial institution. The system provides:

- **Automated process discovery** using Alpha, Heuristic, and Inductive miners
- **Conformance checking** with fitness, precision, simplicity, and generalization metrics
- **Bottleneck analysis** identifying slow transitions in the process
- **LLM-powered reasoning** using ChatGPT for management reports and optimization suggestions
- **Interactive dashboard** for visualization and AI-assisted analysis

## 📁 Project Structure

```
formal-methods/
├── dataset/
│   └── BPI_Challenge_2012.xes    # Original event log (XES format)
├── bpi_results/                   # Generated Petri net visualizations
│   ├── alpha_net.png
│   ├── heuristic_net.png
│   ├── inductive_net_*.png
│   └── bpi2012_comparison*.png
├── mine.py                        # Process discovery & conformance checking
├── reasoning.py                   # LLM reasoning pipeline
├── dashboard.py                   # Streamlit interactive dashboard
├── report.tex                     # LaTeX project report
├── requirements.txt               # Python dependencies
├── eventlog.csv                   # Converted event log (CSV)
├── llm_prompt.txt                 # Generated prompt for LLM
├── llm_response.txt               # LLM analysis response
├── .env                           # Environment variables (API keys)
└── README.md                      # This file
```

## ✨ Features

### 1. Process Discovery (`mine.py`)
- Loads and preprocesses XES event logs
- Applies three discovery algorithms:
  - **Alpha Miner**: Classic algorithm based on ordering relations
  - **Heuristic Miner**: Frequency-based approach robust to noise
  - **Inductive Miner**: Divide-and-conquer ensuring sound models
- Generates Petri net visualizations
- Computes quality metrics (fitness, precision, simplicity, generalization)
- Performs sensitivity analysis on noise thresholds

### 2. LLM Reasoning (`reasoning.py`)
- Extracts process metrics and bottlenecks automatically
- Generates structured prompts for ChatGPT
- Sends context to OpenAI API
- Saves complete analysis with timestamps for traceability

### 3. Interactive Dashboard (`dashboard.py`)
- **Model Results Tab**: View discovered Petri nets
- **Bottleneck Analysis Tab**: Visualize slow transitions
- **AI Analyst Tab**: Chat with GPT for process insights and next-best-action predictions

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- [Graphviz](https://graphviz.org/download/) (required for Petri net visualization)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd formal-methods
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Graphviz

**Windows:**
```bash
# Using Chocolatey
choco install graphviz

# Or download from https://graphviz.org/download/
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install graphviz
```

**macOS:**
```bash
brew install graphviz
```

## ⚙️ Configuration

### OpenAI API Key

Create a `.env` file in the project root with your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

> **Note:** You can get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)

### Dataset

Ensure the BPI Challenge 2012 dataset is placed in `dataset/BPI_Challenge_2012.xes`.

You can download it from [4TU.ResearchData](https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204).

## 📖 Usage

### Step 1: Run Process Mining Pipeline

```bash
python mine.py
```

This will:
- Load and convert the XES file to CSV
- Discover process models using all three algorithms
- Generate Petri net visualizations in `bpi_results/`
- Compute and display quality metrics
- Create comparison charts

### Step 2: Run LLM Reasoning Pipeline

```bash
python reasoning.py
```

This will:
- Extract metrics from the event log
- Identify top bottlenecks
- Generate a structured prompt (`llm_prompt.txt`)
- Send the prompt to ChatGPT API
- Save the complete analysis (`llm_response.txt`)

### Step 3: Launch Interactive Dashboard

```bash
streamlit run dashboard.py
```

This opens a web browser with:
- Process model visualizations
- Bottleneck analysis charts
- AI-powered chat for process analysis
- Next-best-action predictions

## 📊 Output Files

| File | Description |
|------|-------------|
| `eventlog.csv` | Converted event log in CSV format |
| `llm_prompt.txt` | Structured prompt sent to ChatGPT |
| `llm_response.txt` | Complete LLM analysis with metadata |
| `bpi_results/*.png` | Petri net visualizations |
| `bpi_results/bpi2012_comparison*.png` | Metric comparison charts |

## 📄 License

See [LICENSE](LICENSE) file for details.

## 👥 Authors

Formal Methods Course - Academic Year 2024/2025
