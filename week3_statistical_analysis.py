import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

# Load the public Wine Recognition dataset
wine = load_wine(as_frame=True)
df = wine.frame.copy()
df.columns = list(wine.feature_names) + ["target"]
df["class"] = df["target"].map({0: "Class 1", 1: "Class 2", 2: "Class 3"})

# Data quality checks
print("Shape:", df.shape)
print("Missing cells:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# Research question
# Is the mean alcohol content different between Class 1 and Class 2?
class1 = df.loc[df["class"] == "Class 1", "alcohol"]
class2 = df.loc[df["class"] == "Class 2", "alcohol"]
class3 = df.loc[df["class"] == "Class 3", "alcohol"]

# Hypotheses
# H0: mu_class1 = mu_class2
# H1: mu_class1 != mu_class2

# Welch independent-samples t-test
t_result = stats.ttest_ind(class1, class2, equal_var=False)
print("Welch t-test:", t_result)

# 95% CI for the difference in means (Class 1 - Class 2)
mean_diff = class1.mean() - class2.mean()
se = np.sqrt(class1.var(ddof=1)/len(class1) + class2.var(ddof=1)/len(class2))
df_welch = (class1.var(ddof=1)/len(class1) + class2.var(ddof=1)/len(class2))**2 / ((class1.var(ddof=1)/len(class1))**2/(len(class1)-1) + (class2.var(ddof=1)/len(class2))**2/(len(class2)-1))
crit = stats.t.ppf(0.975, df_welch)
ci = (mean_diff - crit*se, mean_diff + crit*se)
print("Mean difference:", mean_diff)
print("95% CI:", ci)

# One-way ANOVA across all three classes
anova = stats.f_oneway(class1, class2, class3)
print("One-way ANOVA:", anova)

# Effect size (eta squared)
grand_mean = df["alcohol"].mean()
ss_between = sum(len(g)*(g.mean()-grand_mean)**2 for g in [class1,class2,class3])
ss_total = ((df["alcohol"]-grand_mean)**2).sum()
print("Eta squared:", ss_between/ss_total)

# Assumption checks
print("Levene test:", stats.levene(class1, class2, class3, center="median"))
for label, group in [("Class 1", class1), ("Class 2", class2), ("Class 3", class3)]:
    print(label, "Shapiro-Wilk:", stats.shapiro(group))

# Post-hoc Tukey HSD after significant ANOVA
print(pairwise_tukeyhsd(df["alcohol"], df["class"], alpha=0.05))

# Save the working dataset
df.to_csv("wine_dataset.csv", index=False)

# Basic supporting visualization
for cls, group in [("Class 1",class1), ("Class 2",class2), ("Class 3",class3)]:
    plt.hist(group, bins=10, alpha=0.45, label=cls)
plt.title("Alcohol Distribution by Wine Class")
plt.xlabel("Alcohol (%)")
plt.ylabel("Frequency")
plt.legend()
plt.show()
