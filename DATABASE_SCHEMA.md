# EMI-Predict AI - Data Schema Documentation

## Dataset Overview

**Source**: `data/emi_prediction_dataset.csv`

**Size**: 404,802 records

**Features**: 27 columns (22 input features + 5 scenario/target columns)

**Tasks**:
1. Classification: Predict `emi_eligibility`
2. Regression: Predict `max_monthly_emi`

## Feature Categories

### 1. Demographic Features (5 features)

| Feature | Type | Description | Values/Range | Missing Values |
|---------|------|-------------|--------------|----------------|
| `age` | Numeric | Age of the applicant | 18-65 years | Possible |
| `gender` | Categorical | Gender of the applicant | Male, Female, Other | No |
| `marital_status` | Categorical | Marital status | Single, Married, Divorced, Widowed | No |
| `education` | Categorical | Highest education level | High School, Graduate, Professional | No |
| `family_size` | Numeric | Number of family members | 1-8 | Possible |

### 2. Employment Features (4 features)

| Feature | Type | Description | Values/Range | Missing Values |
|---------|------|-------------|--------------|----------------|
| `monthly_salary` | Numeric | Monthly gross salary (INR) | 15,000-200,000 | Possible |
| `employment_type` | Categorical | Type of employment | Private, Government, Self-employed | No |
| `years_of_employment` | Numeric | Years in current employment | 0-40 years | Possible |
| `company_type` | Categorical | Company category | Startup, Mid-size, MNC, PSU | Possible |

### 3. Housing Features (3 features)

| Feature | Type | Description | Values/Range | Missing Values |
|---------|------|-------------|--------------|----------------|
| `house_type` | Categorical | Type of residence | Own, Rented, Family | No |
| `monthly_rent` | Numeric | Monthly rent payment (INR) | 0-50,000 (0 if Own/Family) | No |

### 4. Dependent Features (2 features)

| Feature | Type | Description | Values/Range | Missing Values |
|---------|------|-------------|--------------|----------------|
| `dependents` | Numeric | Number of dependents | 0-6 | Possible |
| `school_fees` | Numeric | Monthly school fees (INR) | 0-25,000 | Possible |
| `college_fees` | Numeric | Monthly college fees (INR) | 0-40,000 | Possible |

### 5. Expense Features (4 features)

| Feature | Type | Description | Values/Range | Missing Values |
|---------|------|-------------|--------------|----------------|
| `travel_expenses` | Numeric | Monthly travel costs (INR) | 1,000-15,000 | Possible |
| `groceries_utilities` | Numeric | Monthly groceries & utilities (INR) | 3,000-30,000 | Possible |
| `other_monthly_expenses` | Numeric | Other monthly expenses (INR) | 1,000-20,000 | Possible |

### 6. Financial Status Features (4 features)

| Feature | Type | Description | Values/Range | Missing Values |
|---------|------|-------------|--------------|----------------|
| `existing_loans` | Categorical | Has existing loans | Yes, No | No |
| `current_emi_amount` | Numeric | Current monthly EMI (INR) | 0-50,000 | No |
| `credit_score` | Numeric | Credit score | 300-850 | Possible |
| `bank_balance` | Numeric | Current bank balance (INR) | 5,000-2,000,000 | Possible |
| `emergency_fund` | Numeric | Emergency fund savings (INR) | 0-500,000 | Possible |

### 7. Scenario & Target Features (5 features)

| Feature | Type | Description | Values/Range | Notes |
|---------|------|-------------|--------------|-------|
| `emi_scenario` | Categorical | Type of EMI requested | Personal Loan EMI, Vehicle EMI, Education EMI, Home Loan EMI, E-commerce Shopping EMI | Context for the EMI request |
| `requested_amount` | Numeric | Loan amount requested (INR) | 10,000-5,000,000 | Principal loan amount |
| `requested_tenure` | Numeric | Loan tenure requested (months) | 6-240 months | Loan duration |
| `emi_eligibility` | Categorical | **TARGET 1** (Classification) | Eligible, Not_Eligible | Binary classification target |
| `max_monthly_emi` | Numeric | **TARGET 2** (Regression) | 500-50,000 INR | Maximum affordable monthly EMI |

## Data Statistics

### Numerical Features Summary

| Feature | Mean | Median | Std Dev | Min | Max |
|---------|------|--------|---------|-----|-----|
| age | ~40 | 38 | ~12 | 18 | 65 |
| monthly_salary | ~55,000 | 50,000 | ~25,000 | 15,000 | 200,000 |
| current_emi_amount | ~12,000 | 8,000 | ~10,000 | 0 | 50,000 |
| credit_score | ~680 | 680 | ~80 | 300 | 850 |
| bank_balance | ~300,000 | 250,000 | ~200,000 | 5,000 | 2,000,000 |

### Categorical Features Distribution

**Gender**:
- Male: ~48%
- Female: ~50%
- Other: ~2%

**Education**:
- High School: ~30%
- Graduate: ~45%
- Professional: ~25%

**Employment Type**:
- Private: ~60%
- Government: ~25%
- Self-employed: ~15%

**House Type**:
- Own: ~35%
- Rented: ~45%
- Family: ~20%

**Existing Loans**:
- Yes: ~55%
- No: ~45%

**EMI Eligibility** (Target):
- Eligible: ~60%
- Not_Eligible: ~40%

## Derived Features (Feature Engineering)

These features will be created during preprocessing:

### Financial Health Indicators

1. **total_monthly_expenses**
   ```python
   total_monthly_expenses = (monthly_rent + groceries_utilities + 
                            travel_expenses + other_monthly_expenses +
                            school_fees + college_fees)
   ```

2. **debt_to_income_ratio** (DTI)
   ```python
   DTI = (current_emi_amount + total_monthly_expenses) / monthly_salary
   ```
   - Critical indicator for loan approval
   - Typical threshold: <40% is healthy

3. **expense_ratio**
   ```python
   expense_ratio = total_monthly_expenses / monthly_salary
   ```
   - Percentage of income spent on expenses
   - Lower is better for EMI eligibility

4. **savings_rate**
   ```python
   monthly_savings = monthly_salary - total_monthly_expenses - current_emi_amount
   savings_rate = monthly_savings / monthly_salary
   ```
   - Higher savings rate = better eligibility

5. **financial_stress_index**
   ```python
   monthly_liquidity = monthly_salary + (bank_balance / 12)
   financial_stress = (current_emi_amount + total_monthly_expenses) / monthly_liquidity
   ```
   - Combines income and savings
   - Lower value = less financial stress

6. **credit_utilization_score**
   ```python
   expected_max_emi = monthly_salary * 0.5  # 50% of salary
   credit_utilization = current_emi_amount / expected_max_emi
   ```
   - How much of potential EMI capacity is used

7. **emergency_fund_ratio**
   ```python
   emergency_fund_ratio = emergency_fund / (monthly_salary * 6)
   ```
   - Should ideally be 1.0 (6 months of salary)

8. **dependents_per_income**
   ```python
   dependents_per_income = dependents / (monthly_salary / 10000)
   ```
   - Normalized dependents by income level

### Interaction Features

9. **age_employment_interaction**
   ```python
   age_employment = age * years_of_employment
   ```

10. **salary_credit_score_interaction**
    ```python
    salary_credit = (monthly_salary / 1000) * credit_score
    ```

### Boolean Indicators

11. **has_rent_payment**
    ```python
    has_rent = (monthly_rent > 0)
    ```

12. **has_education_expenses**
    ```python
    has_education_exp = ((school_fees + college_fees) > 0)
    ```

13. **high_credit_score**
    ```python
    high_credit = (credit_score >= 700)
    ```

## Data Quality Issues

### Missing Values

Expected missing value patterns:
- `monthly_salary`: ~2-3%
- `credit_score`: ~5%
- `bank_balance`: ~3%
- `years_of_employment`: ~1%
- `company_type`: ~10% (for self-employed)

**Handling Strategy**:
- Numerical: Median imputation
- Categorical: Mode imputation or "Unknown" category

### Outliers

Features prone to outliers:
- `monthly_salary` (very high earners)
- `bank_balance` (wealthy individuals)
- `requested_amount` (very large loans)

**Handling Strategy**:
- IQR method: Remove values beyond 1.5 × IQR
- Or cap at 99th percentile

### Data Imbalance

- Target 1 (`emi_eligibility`): 60-40 split (moderate imbalance)
  - May require class weights or SMOTE
- Target 2 (`max_monthly_emi`): Continuous, no imbalance

## Feature Encoding Strategy

### Categorical Features Encoding

1. **Binary Encoding** (2 categories):
   - `existing_loans`: {Yes: 1, No: 0}
   - `gender`: One-hot or label encoding

2. **One-Hot Encoding** (3-5 categories):
   - `marital_status`: 4 categories
   - `education`: 3 categories
   - `employment_type`: 3 categories
   - `house_type`: 3 categories
   - `company_type`: 4 categories
   - `emi_scenario`: 5 categories

3. **Label Encoding** (for tree-based models):
   - All categorical features can use label encoding for Random Forest/XGBoost

### Target Encoding

- **Classification Target** (`emi_eligibility`):
  - Eligible → 1
  - Not_Eligible → 0

- **Regression Target** (`max_monthly_emi`):
  - Use as-is (numeric)
  - May apply log transformation if right-skewed

## Feature Scaling

**Method**: StandardScaler (z-score normalization)

**Features to Scale**:
- All numerical features
- Derived numerical features

**Formula**:
```python
z = (x - μ) / σ
```

**Exception**: Boolean features (0/1) don't need scaling

## Data Splits

### Training Split Strategy

- **Training Set**: 70% (~283,361 records)
  - Used for model training
  
- **Validation Set**: 15% (~60,720 records)
  - Used for hyperparameter tuning
  - Early stopping
  
- **Test Set**: 15% (~60,721 records)
  - Final model evaluation
  - Never seen during training

### Stratification

- **Classification Task**: Stratify by `emi_eligibility`
- **Regression Task**: Use quantile-based stratification on `max_monthly_emi`

## Data Versioning

Track dataset versions with:
- File hash (MD5/SHA256)
- Record count
- Feature statistics
- Creation timestamp

Log in MLflow for reproducibility.

## Privacy & Compliance

- **PII**: Age, gender may be considered PII
- **Anonymization**: No names, IDs, or direct identifiers
- **Compliance**: Ensure adherence to local data protection laws
- **Sensitive Data**: Credit scores, financial data handled securely

## Data Quality Checks

Before training, validate:
1. No duplicate records
2. Target variables present and valid
3. No impossible values (negative salary, age > 100, etc.)
4. Consistent data types
5. Expected value ranges
6. Correlation between features (detect multicollinearity)

## Feature Importance (Expected)

Based on domain knowledge, most important features likely:

**For Eligibility**:
1. `credit_score`
2. `monthly_salary`
3. `current_emi_amount`
4. `existing_loans`
5. `debt_to_income_ratio` (derived)

**For EMI Amount**:
1. `monthly_salary`
2. `bank_balance`
3. `credit_score`
4. `savings_rate` (derived)
5. `current_emi_amount`

