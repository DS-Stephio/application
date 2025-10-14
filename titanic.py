"""
Titanic
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns





# os.chdir('/home/coder/work/ensae-reproductibilite-application')
TrainingData = pd.read_csv("data.csv")

TrainingData.head()


TrainingData["Ticket"].str.split("/").str.len()

TrainingData["Name"].str.split(",").str.len()

N_TREES = 20
MAX_DEPTH = None
MAX_FEATURES = "sqrt"

TrainingData.isnull().sum()


## Un peu d'exploration et de feature engineering

### Statut socioéconomique

fig, axes = plt.subplots(
    1, 2, figsize=(12, 6)
)  # layout matplotlib 1 ligne 2 colonnes taile 16*8
fig1_pclass = sns.countplot(data=TrainingData, x="Pclass", ax=axes[0]).set_title(
    "fréquence des Pclass"
)
fig2_pclass = sns.barplot(
    data=TrainingData, x="Pclass", y="Survived", ax=axes[1]
).set_title("survie des Pclass")


### Age

sns.histplot(data=TrainingData, x="Age", bins=15, kde=False).set_title(
    "Distribution de l'âge"
)
plt.show()

## Encoder les données imputées ou transformées.
numeric_features = ["Age", "Fare"]
categorical_features = ["Embarked", "Sex"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder()),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("Preprocessing numerical", numeric_transformer, numeric_features),
        (
            "Preprocessing categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=20)),
    ]
)


# splitting samples
y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")

# Scinder l'échantillon en 2 sous échantillons train et test.
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.
X_TRAIN, X_TEST, y_train, y_test = train_test_split(X, y, test_size=0.1)
pd.concat([X_TRAIN, y_train], axis=1).to_csv("train.csv")
pd.concat([X_TEST, y_test], axis=1).to_csv("test.csv")

JETONAPI = "$trotskitueleski1917"


# Ici demandons d'avoir 20 arbres
pipe.fit(X_TRAIN, y_train)


# calcul du score sur les datasets d'apprentissage et de test
# le score étant le nombre de bonne prédiction
rdmf_score = pipe.score(X_TEST, y_test)
rdmf_score_tr = pipe.score(X_TRAIN, y_train)
print(f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation")


print(20 * "-")
print("matrice de confusion")
print(confusion_matrix(y_test, pipe.predict(X_TEST)))
