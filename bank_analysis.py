
import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv("bank_customers.csv")

output = Path("output")
output.mkdir(exist_ok=True)

print("\n========== BANK CUSTOMER DATA ==========")
print(df.head(10))

print("\nDataset Shape:", df.shape)

df = df.drop_duplicates()

columns = ["Age", "Credit_Score", "Annual_Income", "Account_Balance"]

for col in columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

print("\n========== DATA CLEANING ==========")
print("Missing Values:")
print(df.isnull().sum())

df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0, 30, 45, 60, np.inf],
    labels=["Young", "Middle-aged", "Senior", "Older"]
)

df["Risk_Level"] = np.where(
    df["Credit_Score"] >= 700,
    "Low Risk",
    np.where(
        df["Credit_Score"] >= 650,
        "Medium Risk",
        "High Risk"
    )
)

df["Loan_Eligibility"] = np.where(
    (df["Credit_Score"] >= 680) &
    (df["Annual_Income"] > 400000),
    "Eligible",
    "Not Eligible"
)

df["Customer_Type"] = np.where(
    df["Account_Balance"] > 400000,
    "High Value",
    np.where(
        df["Account_Balance"] >= 100000,
        "Medium Value",
        "Low Value"
    )
)

print("\n========== CUSTOMER ANALYSIS ==========")
print(
    df[
        [
            "Customer_ID",
            "Age_Group",
            "Risk_Level",
            "Loan_Eligibility",
            "Customer_Type"
        ]
    ]
)

high_value = df[df["Account_Balance"] > 400000]

print("\n========== HIGH-VALUE CUSTOMERS ==========")
print(high_value[["Customer_ID", "Account_Balance"]])

print("\n========== FINANCIAL STATISTICS ==========")
print("Total Customers:", len(df))
print("Average Age:", round(df["Age"].mean(), 2))
print("Average Credit Score:", round(df["Credit_Score"].mean(), 2))
print("Average Income:", round(df["Annual_Income"].mean(), 2))
print("Average Balance:", round(df["Account_Balance"].mean(), 2))
print("Total Balance:", df["Account_Balance"].sum())
print("Maximum Balance:", df["Account_Balance"].max())
print("Minimum Balance:", df["Account_Balance"].min())

print("\n========== RISK ANALYSIS ==========")
risk_report = df["Risk_Level"].value_counts()
print(risk_report)

print("\n========== LOAN ELIGIBILITY ==========")
loan_report = df["Loan_Eligibility"].value_counts()
print(loan_report)

print("\n========== CUSTOMER VALUE ANALYSIS ==========")
value_report = df["Customer_Type"].value_counts()
print(value_report)

print("\n========== AGE GROUP ANALYSIS ==========")
age_report = df.groupby("Age_Group", observed=False)[
    ["Annual_Income", "Account_Balance"]
].mean()

print(age_report)

df.to_csv(output / "analyzed_customers.csv", index=False)
high_value.to_csv(output / "high_value_customers.csv", index=False)
risk_report.to_csv(output / "risk_report.csv")
loan_report.to_csv(output / "loan_report.csv")
value_report.to_csv(output / "customer_value_report.csv")
age_report.to_csv(output / "age_group_report.csv")

print("\nAll reports saved successfully!")
print("========== ANALYSIS COMPLETED ==========")
