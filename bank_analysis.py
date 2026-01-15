import pandas as pd
import numpy as np

df = pd.read_csv("bank_customers.csv")

print("\n--- BANK CUSTOMER DATA ---")
print(df)

df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[18, 30, 45, 60],
    labels=["Young", "Middle-aged", "Senior"]
)

df["Risk_Level"] = np.where(
    df["Credit_Score"] >= 700, "Low Risk",
    np.where(df["Credit_Score"] >= 650, "Medium Risk", "High Risk")
)

df["Loan_Eligibility"] = np.where(
    (df["Credit_Score"] >= 680) & (df["Annual_Income"] > 400000),
    "Eligible",
    "Not Eligible"
)

high_value = df[df["Account_Balance"] > 400000]

print("\n--- CUSTOMER ANALYSIS ---")
print(df[["Customer_ID", "Age_Group", "Risk_Level", "Loan_Eligibility"]])

print("\nHigh-Value Customers:")
print(high_value[["Customer_ID", "Account_Balance"]])

print("\nAccount Balance Statistics:")
print("Mean:", np.mean(df["Account_Balance"]))
print("Max:", np.max(df["Account_Balance"]))
print("Min:", np.min(df["Account_Balance"]))
