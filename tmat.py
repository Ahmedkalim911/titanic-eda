import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

def load_data():
    df = sns.load_dataset("titanic")
    return df

def explore_data(df):
    print("Shape:", df.shape)
    print("\nHead:\n", df.head())
    print("\nDescription:\n", df.describe())
    print("\nMissing Values:\n", df.isnull().sum())

def clean_data(df):
    df["age"] = df["age"].fillna(df["age"].median())
    df = df.drop("deck", axis=1)# 
    df = df.dropna(subset=["embark_town"])#
    return df
def visualize_data(df):
  plt.figure(figsize=(12, 10))
  plt.subplot(2, 2, 1)
  sns.barplot(x="class", y="survived", hue="sex",  data=df)
  plt.title("Survival by class and sex")

  plt.subplot(2, 2, 2)
  sns.barplot(x="class", y="survived", data=df)
  plt.title("Survival by Class")

  plt.subplot(2, 2, 3)
  plt.hist(df["age"], bins=50)
  plt.title("Age Distribution")

  plt.subplot(2, 2, 4)
  sns.boxplot(x="survived", y="age", data=df)
  plt.title("Age vs Survival")
 
  plt.tight_layout()
  plt.show()

  plt.figure(figsize=(8, 6))
  sns.heatmap(df.corr(numeric_only=True), annot=True)
  plt.title("Correlation Matrix")

  plt.figure(figsize=(8, 6))
  sns.boxplot(x="survived", y="fare", data=df)
  plt.title("Fare vs Survival")
  plt.show()

  plt.figure(figsize=(8, 6))
  sns.barplot(x="age", y="survived", data=df)
  plt.title("Survival by Age")
  plt.show()

  sns.catplot(x="class", y="survived", hue="sex", col="embark_town", data=df, kind="bar")
  plt.show()
 
def show_insights(df):
    print("\n--- Insights ---")

    survival_by_sex = df.groupby("sex")["survived"].mean()
    print("\nSurvival by Sex:\n", survival_by_sex)

    survival_by_class = df.groupby("class")["survived"].mean()
    print("\nSurvival by Class:\n", survival_by_class)

    print("\nInsight 1: Females have significantly higher survival rates than males.")
    print("Insight 2: First class passengers had the highest survival rate.")
    print("Insight 3: Survival is affected by both class and sex together.")

def main():
    df = load_data()
    explore_data(df)
    df = clean_data(df)
    visualize_data(df)
    show_insights(df)
if __name__ == "__main__":
    main()