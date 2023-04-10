import pandas as pd
import polars as pl
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Data processing 

telco_data = pl.read_csv("/Users/[redacted]/Downloads/telco-customer-churn.csv")

dummy_var = ( 
    telco_data
    .select(pl.exclude([pl.Int64, pl.Float64]))
    .select(pl.exclude("customerID"))
    .columns
)


telco_data = telco_data.to_pandas()
telco_data = pd.get_dummies(telco_data, columns = dummy_var, drop_first = True)

telco_data = (
    pl.from_pandas(telco_data)
    .select(pl.exclude("customerID"))
)

# Target variable - class balance

(telco_data
     .groupby("Churn_Yes")
     .agg(pl.count()/len(telco_data))
)

# Model building 

mod_xgb = XGBClassifier()

X = telco_data.select(pl.exclude("Churn_Yes")).to_numpy()
Y = telco_data.select(pl.col("Churn_Yes")).to_numpy()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 421)

mod_xgb.fit(x_train, y_train)
y_pred = mod_xgb.predict(x_test)

# Model accuracy

print(accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cm_graphic = ConfusionMatrixDisplay(cm)
cm_graphic.plot()

print(classification_report(y_test, y_pred))
