# Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.ticker as mtick

# ============================================
# 1. LOAD DATA
# ============================================
met_file = r"C:\Users\Frank\Desktop\cs\project\datapro\met.xlsx"
met_data = pd.read_excel(met_file, engine='openpyxl')

propdata_file = r"C:\Users\Frank\Desktop\cs\project\datapro\prop.csv"
prop_data = pd.read_csv(propdata_file)
print("Data loaded successfully")

# ============================================
#  ensure price column exists
# ============================================
if 'price' not in prop_data.columns:
    raise KeyError("Column 'price' not found in prop.csv - check your CSV header")

# ============================================
# 2. CLEAN HPI DATA
# ============================================
print(f"Original HPI data shape: {met_data.shape}")
print(f"HPI columns: {list(met_data.columns)}")
print(f"Sample HPI dates before cleaning:\n{met_data['Date'].head()}")

# Clean date column robustly
met_data_clean = met_data.dropna()
print(f"After dropna: {len(met_data_clean)} rows")

# Strip whitespace from date strings
met_data_clean['Date'] = met_data_clean['Date'].astype(str).str.strip()
print(f"Sample dates after strip: {met_data_clean['Date'].head().tolist()}")

# Parse dates, coerce errors to NaT

met_data_clean['Date'] = pd.to_datetime(met_data_clean['Date'], errors='coerce')
print(f"After date parsing: {met_data_clean['Date'].notna().sum()} valid dates")

# Drop rows where date parsing failed

met_data_clean = met_data_clean.dropna(subset=['Date'])
print(f"After dropping invalid dates: {len(met_data_clean)} rows")

# Check if any valid dates remain
if len(met_data_clean) == 0:
    print("\nERROR: All HPI dates failed to parse!")
    print("Check your Excel file 'Date' column format.")
    print("\nFirst few rows of original date column:")
    print(met_data['Date'].head(10))
    raise SystemExit

# Extract year and month for merging
met_data_clean['Year'] = met_data_clean['Date'].dt.year
met_data_clean['Month'] = met_data_clean['Date'].dt.month

# Clean England HPI numeric column robustly
met_data_clean['England'] = (
    met_data_clean['England']
    .astype(str)
    .str.replace('£', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)
met_data_clean['England'] = pd.to_numeric(met_data_clean['England'], errors='coerce')
if met_data_clean['England'].isna().all():
    raise ValueError("HPI 'England' column failed to convert to numeric. Check values.")
met_data_clean = met_data_clean.dropna(subset=['England'])

# ============================================
# 3. CLEAN PROPERTY DATA (price & date)
# ============================================
# Convert dateoftransfer to datetime and drop invalid rows
prop_data_clean = prop_data.dropna(subset=['dateoftransfer']).copy()
prop_data_clean['dateoftransfer'] = pd.to_datetime(prop_data_clean['dateoftransfer'], errors='coerce')
prop_data_clean = prop_data_clean.dropna(subset=['dateoftransfer']).copy()

# Robust price cleaning: strip currency symbols/commas and coerce to numeric
prop_data_clean['price'] = prop_data_clean['price'].astype(str).str.replace('£', '', regex=False).str.replace(',', '', regex=False).str.strip()
prop_data_clean['price'] = pd.to_numeric(prop_data_clean['price'], errors='coerce')

# show extreme values so you can inspect them
print("\nTop 10 raw prices after conversion (desc):")
print(prop_data_clean['price'].nlargest(10).to_string(index=False))

# drop rows where price conversion failed
prop_data_clean = prop_data_clean.dropna(subset=['price']).copy()

prop_data_clean['Sale_Year'] = prop_data_clean['dateoftransfer'].dt.year
prop_data_clean['Sale_Month'] = prop_data_clean['dateoftransfer'].dt.month

# ============================================
# 4. STANDARDIZE PROPERTY TYPES
# ============================================
# Map property types to 'House' or 'Flat', others to 'Other'
def standardize_property_type(prop_type):
    mapping = {
        'D': 'House',
        'S': 'House',
        'T': 'House',
        'B': 'House',
        'F': 'Flat'
    }
    return mapping.get(prop_type, 'Other')

if 'propertytype' in prop_data_clean.columns:
    prop_data_clean['Standardized_Property_Type'] = prop_data_clean['propertytype'].apply(standardize_property_type)
else:
    prop_data_clean['Standardized_Property_Type'] = None

# ============================================
# 5. MERGE PROPERTY DATA WITH HPI
# ============================================
print(f"\nProperty sale years: {prop_data_clean['Sale_Year'].min()} to {prop_data_clean['Sale_Year'].max()}")
print(f"HPI years: {met_data_clean['Year'].min()} to {met_data_clean['Year'].max()}")
print(f"Property sales sample months: {prop_data_clean[['Sale_Year', 'Sale_Month']].head()}")
print(f"HPI sample months: {met_data_clean[['Year', 'Month']].head()}")

prop_data_clean['Sale_Year'] = prop_data_clean['Sale_Year'].astype(int)
prop_data_clean['Sale_Month'] = prop_data_clean['Sale_Month'].astype(int)
met_data_clean['Year'] = met_data_clean['Year'].astype(int)
met_data_clean['Month'] = met_data_clean['Month'].astype(int)

merged_data = pd.merge(
    prop_data_clean,
    met_data_clean[['Year', 'Month', 'England']],
    left_on=['Sale_Year', 'Sale_Month'],
    right_on=['Year', 'Month'],
    how='left'
)

merged_data = merged_data.rename(columns={'England': 'HPI_at_Sale'})
# drop the merge keys to avoid confusion
merged_data = merged_data.drop(columns=['Year', 'Month'])

print(f"Merged data: {len(merged_data)} rows")
print(f"HPI_at_Sale missing: {merged_data['HPI_at_Sale'].isnull().sum()} rows")

# ============================================
# 6. CREATE FINAL DATASET
# ============================================
if 'newbuild' not in merged_data.columns:
    merged_data['newbuild'] = None

final_dataset = merged_data[['numberrooms', 'tfarea', 'Standardized_Property_Type',
                             'HPI_at_Sale', 'newbuild', 'price', 'postcode', 'dateoftransfer']].copy()

# Extract first 2 chars of outward code -> KT22 -> KT
def extract_postcode_area(postcode):
    if pd.isna(postcode):
        return None
    s = str(postcode).strip()
    if s == '':
        return None
    outward = s.split(' ')[0]
    return outward[:2].upper()

final_dataset['Postcode_Area'] = final_dataset['postcode'].apply(extract_postcode_area)

# ============================================
# 7. PREPARE FEATURES (X) AND TARGET (Y)
# ============================================
print(f"\nInitial dataset size: {len(final_dataset)}")
print(f"Missing values:\n{final_dataset[['numberrooms', 'tfarea', 'HPI_at_Sale', 'Standardized_Property_Type', 'Postcode_Area', 'price']].isnull().sum()}")

final_dataset = final_dataset.dropna(subset=['HPI_at_Sale'])
print(f"After dropping rows without HPI match: {len(final_dataset)}")

if len(final_dataset) == 0:
    print("\nERROR: No data remains after merge. The HPI dates don't match property sale dates.")
    raise SystemExit

final_dataset = final_dataset.dropna(subset=['numberrooms', 'tfarea', 'price'])
print(f"After dropping NaN in numeric columns: {len(final_dataset)}")

# Optionally inspect unique postcode groups
print("\nUnique postcode groups (sample):", final_dataset['Postcode_Area'].dropna().unique()[:20])
print("Number of unique postcode groups:", final_dataset['Postcode_Area'].nunique())

features_to_use = ['numberrooms', 'tfarea', 'HPI_at_Sale']

if final_dataset['Standardized_Property_Type'].notna().sum() > 0:
    features_to_use.append('Standardized_Property_Type')
    print(f"Added Standardized_Property_Type - unique values: {final_dataset['Standardized_Property_Type'].unique()}")

if final_dataset['Postcode_Area'].notna().sum() > 0:
    features_to_use.append('Postcode_Area')
    print(f"Added Postcode_Area - unique count: {final_dataset['Postcode_Area'].nunique()}")

model_data = final_dataset[features_to_use + ['price']].copy()
print(f"Model data size before encoding: {len(model_data)}")

#postcode reduced to 2-char region 
categorical_features = [f for f in features_to_use if f in ['Standardized_Property_Type', 'Postcode_Area']]
if categorical_features:
    model_data = pd.get_dummies(model_data, columns=categorical_features, drop_first=True)
    print(f"Model data size after encoding: {len(model_data)}")

# make sure target is numeric
model_data['price'] = pd.to_numeric(model_data['price'], errors='coerce')
model_data = model_data.dropna(subset=['price']).copy()

X_data = model_data.drop(columns=['price'])
y_data = model_data['price']

print(f"\nFeatures being used: {list(X_data.columns)}")
print(f"Total features: {X_data.shape[1]}")
print(f"Final sample size: {len(X_data)}")

# ============================================
# 8. TRAIN TEST SPLIT
# ============================================

# split data into training and testing sets
# 80% training, 20% testing
# set random_state for reproducibility

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
print("\nTrain and test split completed")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# ============================================
# 9. LINEAR REGRESSION MODELING
# ============================================
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel training completed")

y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== Model Performance ===")
print(f"RMSE: £{rmse:,.2f}")
print(f"MAE: £{mae:,.2f}")
print(f"R² Score: {r2:.4f}")

# ============================================
# Decision tree regressor modeling
# ============================================
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
print("\nDecision Tree Model training completed")

y_dt_pred = dt_model.predict(X_test)

dt_rmse = mean_squared_error(y_test, y_dt_pred) ** 0.5
dt_mae = mean_absolute_error(y_test, y_dt_pred)
dt_r2 = r2_score(y_test, y_dt_pred)

print(f"\n=== Decision Tree Model Performance ===")
print(f"RMSE: £{dt_rmse:,.2f}")
print(f"MAE: £{dt_mae:,.2f}")
print(f"R² Score: {dt_r2:.4f}")

# ============================================
# 10. VISUALIZE RESULTS OF LINEAR REGRESSION (formatted axes)
# ============================================
# ensure numeric numpy arrays for plotting
y_test_vals = y_test.values.astype(float)
y_pred_vals = np.array(y_pred).astype(float)

plt.figure(figsize=(10, 6))
plt.scatter(y_test_vals, y_pred_vals, alpha=0.6)
plt.plot([y_test_vals.min(), y_test_vals.max()],
         [y_test_vals.min(), y_test_vals.max()], 'r--', lw=2)

ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("£{x:,.0f}"))
ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("£{x:,.0f}"))

# set sensible limits with a small padding so extreme outliers don't squash the plot
x_min, x_max = y_test_vals.min(), y_test_vals.max()
y_min, y_max = y_pred_vals.min(), y_pred_vals.max()
pad = max((x_max - x_min), (y_max - y_min)) * 0.02
ax.set_xlim(max(0, x_min - pad), x_max + pad)
ax.set_ylim(max(0, min(y_min, x_min) - pad), max(y_max, x_max) + pad)

plt.xlabel("Actual Prices (£)")
plt.ylabel("Predicted Prices (£)")
plt.title("Actual vs Predicted Property Prices (Linear Regression)")
plt.tight_layout()
plt.show()

# Show feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': X_data.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n=== Top 10 Feature Coefficients ===")
print(feature_importance.head(10))

# ============================================
# 11. VISUALIZE DECISION TREE RESULTS (formatted axes)
# ============================================
y_dt_pred_vals = np.array(y_dt_pred).astype(float)

plt.figure(figsize=(10, 6))
plt.scatter(y_test_vals, y_dt_pred_vals, alpha=0.6, color='green')
plt.plot([y_test_vals.min(), y_test_vals.max()],
         [y_test_vals.min(), y_test_vals.max()], 'r--', lw=2)

ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("£{x:,.0f}"))
ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("£{x:,.0f}"))

ax.set_xlim(max(0, x_min - pad), x_max + pad)
ax.set_ylim(max(0, min(y_dt_pred_vals.min(), x_min) - pad), max(y_dt_pred_vals.max(), x_max) + pad)

plt.xlabel("Actual Prices (£)")
plt.ylabel("Predicted Prices (£)")
plt.title("Actual vs Predicted Property Prices (Decision Tree)")
plt.tight_layout()
plt.show()



# ============================================
# 12. Random Forest Regressor Modeling
# ============================================

# For Random Forest, we need to ensure all categorical variables are encoded
label_encoder = LabelEncoder()

 # Create a copy of X_data to encode
X_encoded = X_data.copy()

# Encode any remaining categorical columns
for col in X_encoded.select_dtypes(include=['object']).columns:
    # Only encode if there are non-numeric values
    X_encoded[col] = label_encoder.fit_transform(X_encoded[col].astype(str))
    # This will convert categories to integers
    
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_encoded, y_data, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf_model.fit(X_train_rf, y_train_rf)

print("\nRandom Forest Model training completed")


# Predict and evaluate
y_rf_pred = rf_model.predict(X_test_rf)
rf_rmse = mean_squared_error(y_test_rf, y_rf_pred) ** 0.5
rf_mae = mean_absolute_error(y_test_rf, y_rf_pred)
rf_r2 = r2_score(y_test_rf, y_rf_pred)


# results
print(f"\n=== Random Forest Model Performance ===")
print(f"RMSE: £{rf_rmse:,.2f}")
print(f"MAE: £{rf_mae:,.2f}")
print(f"R² Score: {rf_r2:.4f}")


# ============================================
#  13. Visualize Random Forest Results
# ============================================

# ensure numeric numpy arrays for plotting
y_rf_pred_vals = np.array(y_rf_pred).astype(float)
plt.figure(figsize=(10, 6))
plt.scatter(y_test_vals, y_rf_pred_vals, alpha=0.6, color='orange')
plt.plot([y_test_vals.min(), y_test_vals.max()],
         [y_test_vals.min(), y_test_vals.max()], 'r--', lw=2)
ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("£{x:,.0f}"))
ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("£{x:,.0f}"))
ax.set_xlim(max(0, x_min - pad), x_max + pad)
ax.set_ylim(max(0, min(y_rf_pred_vals.min(), x_min) - pad), max(y_rf_pred_vals.max(), x_max) + pad)
plt.xlabel("Actual Prices (£)")
plt.ylabel("Predicted Prices (£)")
plt.title("Actual vs Predicted Property Prices (Random Forest)")
plt.tight_layout()
plt.show()


# ============================================
# 14. COMPARISON OF MODELS
# ============================================
print("\n=== Comparison of Models ===")
comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree Regressor' , 'Random Forest Regressor'],
    'RMSE': [rmse, dt_rmse , rf_rmse],
    'MAE': [mae, dt_mae , rf_mae],
    'R² Score': [r2, dt_r2 , rf_r2]
})
print(comparison_df)
