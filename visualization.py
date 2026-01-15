import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("bank_customers.csv")
risk_counts = df["Risk_Level"].value_counts()

risk_counts.plot(kind="bar", title="Customer Risk Profile")
plt.xlabel("Risk Level")
plt.ylabel("Number of Customers")
plt.show()
