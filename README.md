# Paper C: KDMiLe | Prediction


## 📌 Overview


> This implementation predicts infant mortality using a dataset obtained from probabilistic entity matching of SINASC and SIM databases. The data has its columns and rows filtered, values are scaled with StandardScaler, then SMOTE is applied to balance classes. To perform the prediction, Logistic Regression, XGBoost, Decision Tree and Support Vector are used and compared. 


**"Prediction of Infant Mortality in Brazil using Machine Learning and Entity Matching on Brazilian Unified Health System's Data"**


Authors: Morsoleto, R. et al.
Accepted at: [KDMiLe](https://sbbd.org.br/2025/kdmile/?lang=pt) 2025,
waiting publication.


## 🚀 Setup


To ensure reproduction of results, requirements are listed on [pyproject](pyproject.toml) file. [Poetry](https://python-poetry.org/) can be used to download requirements listed and run [main.py](main.py) with the following commands respectively:


```bash
poetry install
```


```bash
poetry run python main.py
```


## 📝 License
[LGNU](LICENSE) | © GOPAD 2025
