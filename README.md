# Milestone 3: Feature Engineering and Model Expansion for SMS Spam Detection

This project extends the SMS spam detection reproduction from Milestone 2 by introducing **Bigram N-grams**, **Structural Feature Engineering**, and **Model Diversification** to improve performance beyond the original paper's baseline.

The `milestone3.py` script runs a complete **Ablation Study** in three phases:
1.  **Baseline:** Original reproduction (Unigram TF-IDF).
2.  **Experiment 1:** Adding Bigrams (NLP Context).
3.  **Experiment 2:** Adding Structural Features (`Message Length`, `Digit Count`) and Logistic Regression.

---

## How to Reproduce

### 1. Clone the Repository

```bash
git clone [https://github.com/bradenseidl/CAI-Milestone-3.git](https://github.com/bradenseidl/CAI-Milestone-3.git)
cd CAI-Milestone-3
```

---

### 2. Environment Setup

**Create and activate a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

**Install required libraries:**
```bash
pip install -r requirements.txt
```

---

### 3. Dataset

**Download the dataset from the UCI Machine Learning Repository:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection

**Setup the data folder:**
```bash
mkdir data
```

Place the file `SMSSpamCollection` inside the `data/` directory.

---

### 4. Run the Experiment

The code will automatically download NLTK tokenizer data (`punkt`) if missing.

```bash
python milestone3.py
```

---

### 5. Output

- The script prints **Ablation Study** metrics (**Accuracy**, **Precision**, **Recall**, **F1**) for all three experimental phases.  
- Includes 5-fold cross-validation results for the best models.  
- Displays Confusion Matrix comparisons to visualize the reduction in false negatives.  
- Generates **ROC** and **Precision–Recall** plots for the final extended ensemble.

All plots are automatically saved to a `reports/` folder.

```bash
mkdir reports   # only needed once
```

---

## Reproducibility Notes

- **Random Seed:** `RANDOM_STATE = 42` ensures deterministic splits and results.  
- **Libraries:** see `requirements.txt` for full environment details.  
- **Outputs:** ROC and PR curves saved in `/reports`.  

---

**Methodology Note:**
The code implements custom feature engineering by stacking dense numerical features (`Message Length`, `Digit Count`) horizontally with the sparse TF-IDF matrix. `ConvergenceWarning` logs for linear models in Experiment 2 are expected due to unscaled features, which highlights the robustness of the tree-based models in the final report.