import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#ste2: load data
df=pd.read_csv(r"Telco-Customer-Churn.csv")

#step3: overview of data
print("dataset information")
print(df.head())
print(df.info())

#replacing blanks with 0 as tenure is 0 and no total charges are recorded
print(df["TotalCharges"] == df["TotalCharges"].replace(" ","0"))
df["TotalCharges"] = df["TotalCharges"].astype("float")
print(df.info())

print(df.isnull().sum().sum())

print(df.describe())

print(df["customerID"].duplicated().sum())

def conv(value):
    if value == 1:
        return "yes"
    else:
        return "no"

df['SeniorCitizen'] = df["SeniorCitizen"].apply(conv)

# Set plot style
sns.set_style("whitegrid")

#1. Univariate Analysis

# Figure 1: Distribution of Tenure
plt.figure(figsize=(6, 4))
sns.histplot(df["tenure"], bins=30, kde=True, color="#1f77b4")
plt.title("Distribution of Tenure")
plt.xlabel("Tenure (Months)")
plt.ylabel("Count")
plt.show()

# Figure 2: Distribution of Monthly Charges
plt.figure(figsize=(6, 4))
sns.histplot(df["MonthlyCharges"], bins=30, kde=True, color="#ff7f0e")
plt.title("Distribution of Monthly Charges")
plt.xlabel("Monthly Charges ($)")
plt.ylabel("Count")
plt.show()

# Figure 3: Distribution of Total Charges
plt.figure(figsize=(6, 4))
sns.histplot(df["TotalCharges"], bins=30, kde=True, color="#2ca02c")
plt.title("Distribution of Total Charges")
plt.xlabel("Total Charges ($)")
plt.ylabel("Count")
plt.show()

# Figure 4: Count of Contract Types
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="Contract", data=df, hue="Contract", palette="coolwarm", legend=False)
ax.bar_label(ax.containers[0])
plt.title("Count of Contract Types")
plt.xticks(rotation=45)
plt.show()

# Figure 5: Count of Internet Service Types
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="InternetService", data=df, hue="InternetService", palette="coolwarm", legend=False)
ax.bar_label(ax.containers[0])
plt.title("Count of Internet Service Types")
plt.xticks(rotation=45)
plt.show()

# Figure 6: Count of Payment Methods
plt.figure(figsize=(7, 4))
ax = sns.countplot(x="PaymentMethod", data=df, hue="PaymentMethod", palette="coolwarm", legend=False)
ax.bar_label(ax.containers[0])
plt.title("Count of Payment Methods")
plt.xticks(rotation=45)
plt.show()

# Figure 7: Churn Distribution (Count Plot)
ax = sns.countplot(x="Churn", data=df, hue="Churn", palette="coolwarm", legend=False)
ax.bar_label(ax.containers[0])
plt.title("Count of Customers by Churn")
plt.show()

# Figure 8: Churn Percentage (Pie Chart)
plt.figure(figsize = (3,4))
gb = df.groupby("Churn").agg({'Churn':"count"})
plt.pie(gb['Churn'], labels = gb.index, autopct = "%1.2f%%")
plt.title("Percentage of Churned Customeres", fontsize = 10)
plt.show()


#Bivariate Analysis

sns.set_palette("muted")

# Figure 1: Correlation Heatmap
plt.figure(figsize=(6, 4))
corr_matrix = df[["tenure", "MonthlyCharges", "TotalCharges"]].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Figure 2: Monthly Charges vs. Churn (Boxplot)
plt.figure(figsize=(6, 4))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df, hue="Churn", palette=["#1f77b4", "#ff7f0e"], legend=False)
plt.title("Monthly Charges vs. Churn")
plt.show()

# Figure 3: Total Charges vs. Churn (Boxplot)
plt.figure(figsize=(6, 4))
sns.boxplot(x="Churn", y="TotalCharges", data=df, hue="Churn", palette=["#1f77b4", "#ff7f0e"], legend=False)
plt.title("Total Charges vs. Churn")
plt.show()

# Figure 4: Churn by Gender
plt.figure(figsize = (3,3))
sns.countplot(x = "gender", data = df, hue = "Churn")
plt.title("Churn by Gender")
plt.show()

# Figure 5: Churn by Senior Citizen Status
total_counts = df.groupby('SeniorCitizen')['Churn'].value_counts(normalize=True).unstack() * 100

# Plot
fig, ax = plt.subplots(figsize=(4, 4))  # Adjust figsize for better visualization

# Plot the bars
total_counts.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e'])  # Customize colors if desired

# Add percentage labels on the bars
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.text(x + width / 2, y + height / 2, f'{height:.1f}%', ha='center', va='center')

plt.title('Churn by Senior Citizen (Stacked Bar Chart)')
plt.xlabel('SeniorCitizen')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(title='Churn', bbox_to_anchor = (0.9,0.9))  # Customize legend location

plt.show()


# Figure 6: Churn by Contract Type
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="Contract", data=df, hue="Churn", palette=["#1f77b4", "#ff7f0e"])
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
plt.title("Churn by Contract Type")
plt.xticks(rotation=45)
plt.show()

# Figure 7: Tenure Distribution with Churn
plt.figure(figsize=(8, 4))
sns.histplot(df, x="tenure", hue="Churn", bins=30, kde=True, palette=["#1f77b4", "#ff7f0e"])
plt.title("Tenure Distribution with Churn")
plt.show()

# Figure 8: Churn by Payment Method
plt.figure(figsize=(7, 4))
ax = sns.countplot(x="PaymentMethod", data=df, hue="Churn", palette=["#1f77b4", "#ff7f0e"])
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
plt.title("Churn by Payment Method")
plt.xticks(rotation=45)
plt.show()


