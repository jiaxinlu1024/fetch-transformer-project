# **Fetch Transformer Project**

## **1. Project Structure**
```
fetch_transformer_project/
│-- data/                      # Contains datasets (raw & processed)
│   ├── processed/             # Preprocessed datasets (cached for fast loading)
│   ├── raw/                   # Raw datasets in XLSX format
│-- datasets/                   # Custom dataset loaders and iterators
│-- models/                     # Model architectures (sentence transformer, multi-task transformer)
│-- utils/                      # Utility functions for data processing, training, and demos
│-- main.py                      # Main entry point for running the project
│-- Fetch Exercise.ipynb         # Jupyter Notebook with detailed explanations and results
│-- README.md                    # Project documentation
│-- requirements.txt              # Dependencies for running the project
```

---

## **2. Project Overview**
This project implements a **multi-task sentence Transformer** for **classification and named entity recognition (NER)**. The implementation can be reviewed in two different ways:

- **Jupyter Notebook (`Fetch Exercise.ipynb`)**  
  A detailed, step-by-step implementation with explanations. Useful for understanding the logic and theory behind each step.
  
- **Python Script (`main.py`)**  
  A structured, modular implementation that runs the full pipeline. Ideal for quick navigation and execution.

Both approaches achieve the same results. The notebook provides richer explanations, while the script is cleaner and more modular.

---

## **3. Installation and Running the Project**
### **Prerequisites**
- Python 3.10+
- PyTorch (GPU recommended for training)
- Other dependencies listed in `requirements.txt`

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/jiaxinlu1024/fetch-transformer-project.git
   cd fetch-transformer-project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **Running the Project**
#### **Option 1: Running the Notebook**
1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `Fetch Exercise.ipynb` and execute all cells sequentially.

#### **Option 2: Running the Python Script**
1. Execute the script:
   ```bash
   python main.py
   ```
2. The script runs the entire pipeline, with outputs displayed in the console.