import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("titanic.csv")

print("Dataset Loaded Successfully")

# Check missing values
print(df.isnull().sum())

# Fill missing Age values
df["Age"] = df["Age"].fillna(df["Age"].median())

# Remove duplicate rows
df = df.drop_duplicates()

print("Data Cleaning Completed")

# Histogram
plt.hist(df["Age"])
plt.title("Age Distribution")
plt.show()

# Bar Chart
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# Box Plot
sns.boxplot(y=df["Age"])
plt.title("Age Box Plot")
plt.show()

# Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()