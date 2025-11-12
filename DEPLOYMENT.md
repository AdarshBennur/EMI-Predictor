# EMI-Predict AI - Deployment Guide

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Local Deployment](#local-deployment)
4. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)
7. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Overview

This guide walks you through deploying the EMI-Predict AI application to **Streamlit Cloud** for production use with public URL access.

### Deployment Architecture

```
GitHub Repository
    ↓
Streamlit Cloud
    ↓
    ├── Data Loading (400K records)
    ├── Model Serving (6 ML models)
    ├── MLflow Integration
    └── Multi-page Streamlit App
```

---

## Prerequisites

### Required Accounts

1. **GitHub Account** - For version control and deployment source
2. **Streamlit Cloud Account** - Free tier available at [streamlit.io/cloud](https://streamlit.io/cloud)
3. **Git** installed locally

### Required Files Checklist

- ✅ `requirements.txt` - All Python dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `.streamlit/secrets.toml.example` - Secrets template (DO NOT commit actual secrets)
- ✅ `data/emi_prediction_dataset.csv` - Dataset (400K records)
- ✅ Pre-trained models in `models/` directory
- ✅ `.gitignore` - Properly configured

---

## Local Deployment

### Step 1: Environment Setup

```bash
# Clone repository (if not already)
git clone <your-repo-url>
cd "3. EMI-Predict-AI"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy secrets template
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# Edit secrets.toml with your local settings
# (Use default values for local development)
```

### Step 3: Train Models (Optional)

```bash
# Train all models (takes 10-30 minutes depending on hardware)
python3 -c "
from emipredict.data.loader import load_and_preprocess_data
from emipredict.features.engineering import engineer_features
from emipredict.models.classification import train_all_classification_models
from emipredict.models.regression import train_all_regression_models

# Load data
df = load_and_preprocess_data('data/emi_prediction_dataset.csv')
df = engineer_features(df)

# Train models
train_all_classification_models()
train_all_regression_models()
"
```

### Step 4: Run Application

```bash
# Start Streamlit app
streamlit run emipredict/app/main.py

# Access at: http://localhost:8501
```

### Step 5: View MLflow Dashboard

```bash
# In a separate terminal
mlflow ui

# Access at: http://localhost:5000
```

---

## Streamlit Cloud Deployment

### Step 1: Prepare Repository

```bash
# Ensure all changes are committed
git add .
git commit -m "Prepare for Streamlit Cloud deployment"

# Push to GitHub
git push origin main
```

### Step 2: Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**

2. **Click "New app"**

3. **Configure deployment:**
   - **Repository:** Select your GitHub repository
   - **Branch:** `main`
   - **Main file path:** `emipredict/app/main.py`
   - **App URL:** Choose your custom subdomain (e.g., `emi-predict-ai`)

4. **Advanced settings:**
   - **Python version:** 3.9 or 3.10
   - **Secrets:** Copy contents from `.streamlit/secrets.toml.example` and customize

5. **Click "Deploy"**

### Step 3: Configure Secrets in Streamlit Cloud

In the Streamlit Cloud dashboard:

1. Go to **App Settings** → **Secrets**
2. Paste your secrets configuration (based on `secrets.toml.example`)
3. Update values as needed for production
4. Save changes

### Step 4: Monitor Deployment

- Deployment typically takes 5-10 minutes
- Watch logs for any errors
- Once complete, your app will be live at: `https://<your-app-name>.streamlit.app`

---

## Performance Optimization

### Data Loading Optimization

```python
# In emipredict/config/settings.py

class Config:
    # Limit initial data load for faster startup
    MAX_ROWS_INITIAL = 50000  # Load subset for UI
    ENABLE_CACHING = True  # Cache expensive operations
    CACHE_TTL = 3600  # 1 hour cache
```

### Model Size Optimization

```bash
# Compress models (if too large)
python3 -c "
import joblib
import gzip

# Compress each model
for model_file in ['xgboost_classification.pkl', ...]:
    model = joblib.load(f'models/{model_file}')
    with gzip.open(f'models/{model_file}.gz', 'wb') as f:
        joblib.dump(model, f, compress=('gzip', 3))
"
```

### Caching Strategy

```python
# Use Streamlit caching
import streamlit as st

@st.cache_data(ttl=3600)
def load_data():
    """Cache data loading for 1 hour."""
    return pd.read_csv('data/emi_prediction_dataset.csv')

@st.cache_resource
def load_model(model_path):
    """Cache model loading (persists across sessions)."""
    return joblib.load(model_path)
```

---

## Troubleshooting

### Common Issues

#### 1. Large File Upload Errors

**Problem:** Dataset or models too large for Streamlit Cloud

**Solution:**
```bash
# Use Git LFS for large files
git lfs install
git lfs track "*.csv"
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

#### 2. Memory Errors

**Problem:** App crashes due to insufficient memory

**Solution:**
- Reduce `MAX_ROWS_INITIAL` in config
- Use data chunking for processing
- Enable aggressive caching
- Request resource upgrade from Streamlit Cloud

#### 3. Slow Model Loading

**Problem:** App takes too long to start

**Solution:**
```python
# Lazy load models
@st.cache_resource
def get_model(model_name):
    """Load model only when needed."""
    model_path = Config.get_model_save_path(model_name, task)
    return joblib.load(model_path)
```

#### 4. Module Import Errors

**Problem:** `ModuleNotFoundError` on deployment

**Solution:**
```bash
# Ensure all dependencies in requirements.txt
pip freeze > requirements.txt

# Check for missing packages
grep -i "modulenotfounderror" <deployment_logs>
```

### Debug Mode

Enable debug logging in production:

```python
# In .streamlit/secrets.toml
[app]
debug = true
```

---

## Monitoring & Maintenance

### Health Checks

```python
# Add to main.py
def check_system_health():
    """Check critical components."""
    checks = {
        'data_loaded': check_data(),
        'models_available': check_models(),
        'mlflow_connected': check_mlflow()
    }
    return all(checks.values())
```

### Performance Metrics

Monitor these key metrics:

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| App Load Time | < 10s | > 30s |
| Prediction Latency | < 2s | > 10s |
| Memory Usage | < 1GB | > 2GB |
| Error Rate | < 1% | > 5% |

### Regular Maintenance Tasks

**Weekly:**
- Check application logs for errors
- Monitor user feedback
- Verify prediction accuracy

**Monthly:**
- Update dependencies (`pip install --upgrade`)
- Review and clean up old experiments in MLflow
- Check model performance metrics

**Quarterly:**
- Retrain models with new data
- Update documentation
- Performance optimization review

### Model Updates

To update models in production:

```bash
# 1. Train new models locally
python3 scripts/train_models.py

# 2. Test new models
pytest tests/test_models.py

# 3. Commit and push
git add models/
git commit -m "Update models - version 2.0"
git push origin main

# 4. Streamlit Cloud will auto-redeploy
```

---

## Security Best Practices

### 1. Never Commit Secrets

```bash
# Ensure in .gitignore
.streamlit/secrets.toml
.env
*.key
*.pem
```

### 2. Use Environment Variables

```python
# Access secrets safely
import streamlit as st

mlflow_uri = st.secrets["mlflow"]["tracking_uri"]
```

### 3. Input Validation

```python
# Validate all user inputs
def validate_income(income):
    if income < 0 or income > 10_000_000:
        raise ValueError("Invalid income range")
    return income
```

### 4. Rate Limiting

Consider implementing rate limiting for production:

```python
from functools import wraps
import time

def rate_limit(max_calls, time_window):
    """Simple rate limiting decorator."""
    calls = []
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [c for c in calls if c > now - time_window]
            
            if len(calls) >= max_calls:
                st.error("Too many requests. Please try again later.")
                return None
            
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

---

## Cost Optimization

### Streamlit Cloud Free Tier

- **Included:** 1 app, 1GB RAM, public repos
- **Limitations:** Shared resources, no custom domain
- **Best for:** POCs, demos, small user base

### Streamlit Cloud Pro

- **Cost:** ~$250/month (as of 2025)
- **Benefits:** Private repos, more resources, priority support
- **Best for:** Production apps with moderate traffic

### Alternative: Self-Hosting

If you need more control:

```bash
# Deploy on AWS EC2, Google Cloud, or Azure
# Use Docker for containerization
# Nginx for reverse proxy
# SSL certificate for HTTPS
```

---

## Support & Resources

### Documentation

- **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)
- **MLflow Docs:** [mlflow.org/docs](https://mlflow.org/docs)
- **Project Docs:** See `ARCHITECTURE.md`, `DEVGUIDE.md`

### Community

- **Streamlit Forum:** [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues:** Create issues in your repository

### Contact

For project-specific questions:
- **Email:** your-email@example.com
- **GitHub:** @your-github-username

---

## Deployment Checklist

Before going live:

- [ ] All tests passing (`pytest`)
- [ ] Models trained and validated
- [ ] Documentation updated
- [ ] `.gitignore` properly configured
- [ ] Secrets configured (not committed)
- [ ] GitHub repository ready
- [ ] Streamlit Cloud account created
- [ ] App deployed and accessible
- [ ] Performance tested
- [ ] Monitoring setup
- [ ] Team trained on maintenance

---

## Next Steps After Deployment

1. **Share Your App:**
   - Share the public URL with stakeholders
   - Add URL to README.md and project documentation

2. **Collect Feedback:**
   - Set up user feedback mechanism
   - Monitor usage patterns

3. **Iterate:**
   - Implement requested features
   - Optimize based on real-world usage
   - Plan for scaling if needed

---

**Happy Deploying! 🚀**

For detailed architecture and development guidelines, see:
- `ARCHITECTURE.md` - System design and data flow
- `DEVGUIDE.md` - Development best practices
- `SETUP.md` - Detailed setup instructions

