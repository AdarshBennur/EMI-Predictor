# Quick Start Guide - EMI-Predict AI

**Get up and running in 5 minutes!**

---

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Terminal/Command Line access

---

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required Python packages including:
- scikit-learn, XGBoost (ML models)
- MLflow (experiment tracking)
- Streamlit (web interface)
- pandas, numpy (data processing)

---

## Step 2: Train Models (50K subset - ~10 minutes)

```bash
python scripts/train_models.py
```

**What this does:**
- Loads 50,000 records from the dataset for quick validation
- Trains 3 classification models (Logistic Regression, Random Forest, XGBoost)
- Trains 3 regression models (Linear Regression, Random Forest, XGBoost)
- Saves all models to `models/` directory
- Creates MLflow experiments in `mlruns/`

**Expected output:**
```
EMI-PREDICT AI - MODEL TRAINING PIPELINE
Training on SUBSET (50,000 rows) - Quick validation mode
...
✅ ALL MODELS TRAINED SUCCESSFULLY!
```

**Time:** Approximately 10-15 minutes depending on your system

---

## Step 3: Run Streamlit App

```bash
streamlit run emipredict/app/main.py
```

The application will open automatically in your default web browser at:
**http://localhost:8501**

---

## Step 4: Test Features

Navigate through the application and test:

### 📊 **Data Explorer**
- View dataset statistics and distributions
- Use the sidebar to select row count (10K/50K/100K/All)
- Explore features and relationships

### 🎯 **Eligibility Prediction**
- Enter applicant financial details
- Get 3-class prediction:
  - ✅ Eligible (Low risk)
  - ⚠️ High_Risk (Marginal case)
  - ❌ Not_Eligible (High risk)
- View probability distribution for all classes
- Get class-specific recommendations

### 💰 **EMI Amount Prediction**
- Enter applicant details
- Predict maximum affordable monthly EMI
- View financial analysis

### 📈 **Admin Monitoring**
- View trained models
- Check MLflow experiments
- Monitor system status

---

## View MLflow UI (Optional)

To view detailed experiment tracking:

```bash
mlflow ui
```

Then visit: **http://localhost:5000**

**Features:**
- Compare model performance
- View hyperparameters
- Inspect confusion matrices
- Track metrics across runs

---

## Troubleshooting

### Issue: "Model file not found"
**Solution:** Make sure you ran Step 2 to train the models
```bash
python scripts/train_models.py
```

### Issue: "Dataset not found"
**Solution:** Ensure `data/emi_prediction_dataset.csv` exists in the project root

### Issue: Port 8501 already in use
**Solution:** Stop any existing Streamlit instances or use a different port:
```bash
streamlit run emipredict/app/main.py --server.port=8502
```

### Issue: Import errors
**Solution:** Reinstall dependencies:
```bash
pip install -r requirements.txt --force-reinstall
```

---

## Retrain on Full Dataset (Production)

For production use with the complete 400K+ dataset:

1. **Edit training script:**
   ```bash
   # Open scripts/train_models.py
   # Change line: TRAINING_ROWS = 50000
   # To: TRAINING_ROWS = None
   ```

2. **Run training:**
   ```bash
   python scripts/train_models.py
   ```

**Time:** 30-60 minutes for full dataset

**Benefits:**
- Higher accuracy (target: 90%+)
- Better generalization
- Production-ready models

---

## Next Steps

### For Development:
- Explore the codebase in `emipredict/` directory
- Run tests: `pytest tests/`
- Read `DEVGUIDE.md` for development guidelines

### For Deployment:
- Read `DEPLOYMENT.md` for Streamlit Cloud deployment
- Configure `.streamlit/secrets.toml` for production
- Set up CI/CD pipeline

### For Learning:
- Read `ARCHITECTURE.md` for system design
- Explore `DATABASE_SCHEMA.md` for data structure
- Check `SETUP.md` for detailed setup instructions

---

## Success Checklist

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Models trained (`python scripts/train_models.py`)
- [ ] App running (`streamlit run emipredict/app/main.py`)
- [ ] Data Explorer shows 404,800+ total records
- [ ] Eligibility Prediction working with 3-class output
- [ ] EMI Amount Prediction working
- [ ] Admin Monitoring shows trained models
- [ ] MLflow experiments visible (optional)

---

## Quick Commands Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (50K subset)
python scripts/train_models.py

# Run Streamlit app
streamlit run emipredict/app/main.py

# Run MLflow UI (optional)
mlflow ui

# Run tests
pytest tests/

# Check test coverage
pytest tests/ --cov=emipredict

# Format code
black emipredict/

# Lint code
flake8 emipredict/
```

---

## Support

For issues or questions:
- Check `ARCHITECTURE.md`, `SETUP.md`, and `DEVGUIDE.md`
- Review error messages in the Streamlit app
- Check MLflow logs for training issues
- Inspect `logs/emipredict.log` for detailed logs

---

**Happy Predicting! 🚀**

*For detailed documentation, see README.md and other documentation files in the project root.*

