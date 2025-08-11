import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from dev.prediction.utils import pipeline


def prediction(df):
    columns = [
        "APGAR5", "CONSULTAS", "GESTACAO", "PESO", "MESPRENAT",
        "IDADEMAE", "ESTCIVMAE", "PARTO", "RACACOR",
        "ESCMAE2010", "CODOCUPMAE", "CODMUNNASC"]
    scale_X = StandardScaler()
    X = pd.DataFrame(scale_X.fit_transform(df.drop(["OBITO"], axis=1), ),
                     columns=columns)
    y = df["OBITO"]

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.7, random_state=42)

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Logistic Regression
    lr_model = LogisticRegression()
    pipeline(lr_model, X_train, y_train, X_test, y_test)

    # eXtreme Gradient Boost
    xgb = xgboost.XGBClassifier()
    pipeline(xgb, X_train, y_train, X_test, y_test)

    # Decision Tree
    dec_tree = DecisionTreeClassifier()
    pipeline(dec_tree, X_train, y_train, X_test, y_test)

    # Support Vector Classifier
    svc = SVC(kernel='rbf', gamma=0.5, C=1.0)
    pipeline(svc, X_train, y_train, X_test, y_test)
