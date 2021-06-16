import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

file = './data/abalone_data.csv'
column_names = ['Sex', 'Length_mm', 'Diameter_mm', 'Height_mm', 'Whole_weight_gms',
                'Shucked_weight_gms', 'Viscera_weight_gms', 'Shell_weight_gms', 'Rings']


def load_csv():
    """
    dataframe = load_csv()
    :param file: file location of csv data
    :return: Dataframe
    """
    df = pd.read_csv(file, header=None, names=column_names)

    df['Age'] = df['Rings'] + 1.5
    df['AgeClass'] = pd.cut(df['Age'], bins=[0, 7, 10, 15, np.inf],
                            labels=[1, 2, 3, 4]).astype('category')

    df['Sex'] = pd.Categorical(df['Sex'])
    df['Sex'] = df.Sex.cat.codes

    print("From original csv, added 'Age', 'AgeClass', and changed 'Sex' to category")

    return df


def save_model(model, save_as):
    """
    Save a tensorflow model
    :param model: tensorflow model
    :param save_as: string
    :return:
    """
    model.save(save_as)


def load_model(saved_as):
    """
    Re-load a saved tensorflow model
    :param saved_as: string name
    :return: tensorflow model
    """
    model = tf.keras.models.load_model(saved_as)
    return model


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.savefig('./Images/ROC_y_train_y_scores.png')
    # plt.show()


def plot_feature_importance(df, model):
    n_features = df.shape[1]
    plt.figure(figsize=(12, 8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), df.columns)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.savefig('./Images/Feature_importance.png')
    # plt.show()


# Training Set Size on Error Stability

def plot_learning_curves(model, X, y, num_samples):
    #X_train, X_test, X_valid, y_train, y_test, y_valid = train_test_valid(X, y)
    X_train, X_test, y_train, y_test, = train_test(X, y)
    train_errors, test_errors = [], []
    #valid_errors = []
    for m in range(1, int(num_samples)):
        model.fit(X_train[:m], y_train[:m])
        y_predict = model.predict(X_train[:m])
        y_test_predict = model.predict(X_test)
        #y_valid_predict = model.predict(X_valid)
        train_errors.append(mean_squared_error(y_train[:m], y_predict))
        test_errors.append(mean_squared_error(y_test, y_test_predict))
        #val_errors.append(mean_squared_error(y_valid, y_valid_predict))

    plt.plot(np.sqrt(train_errors), "r-+", lw=1, label="train")
    plt.plot(np.sqrt(test_errors), "b--", lw=2, label="test")
    #plt.plot(np.sqrt(val_errors), "g-", lw=2, label='validation')


def train_test_valid(X, y):
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, test_size=0.2,  random_state=42)
    return X_train, X_test, X_valid, y_train, y_test, y_valid


def train_test(X, y, test_size=0.4):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42) #
    return X_train, X_test, y_train, y_test
