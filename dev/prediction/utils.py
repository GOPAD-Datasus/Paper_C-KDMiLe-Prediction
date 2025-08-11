from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFE


def result_report(y_test, y_pred):
    print("Classification Report")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def pipeline(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result_report(y_test, y_pred)

    rfe_method = RFE(
        model,
        n_features_to_select=10,
        step=2
    )
    rfe_method.fit(X_train, y_train)

    X_train_select = rfe_method.transform(X_train)
    y_train_select = rfe_method.transform(y_train)

    model.fit(X_train_select, y_train_select)
    y_pred = model.predict(X_test)

    result_report(y_test, y_pred)
