import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def model_train(X_train, y_train, filename):
    rf_model = RandomForestClassifier(max_depth = 4, random_state = 100)
    rf_model.fit(X_train, y_train)

    rf_Features = dict(zip(X_train.columns, rf_model.feature_importances_))
    rf_Importance = sorted(rf_Features, key = rf_Features.get, reverse = True)
    rf_Importance = {i:rf_Features[i] for i in rf_Importance}
    print(rf_Importance)

    fig_path = os.path.join(os.getcwd(), "importance.pdf")
    pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)

    fig = plt.figure()
    plt.xlabel('Relative Importance')

    for key, value in rf_Importance.items():
        plt.barh(key, value, height = 0.5, color = 'b', align='center')

    plt.rcParams.update({'font.size': 12})
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    pdf.close()


    # example_tree = rf_model.estimators_[0]
    # labels = list(rf_Features.keys())
    # my_tree = Source(tree.export_graphviz(example_tree, out_file = None, feature_names = labels, class_names = ['Low', 'High'], filled = True))
    # my_tree.format = 'png'
    # my_tree.render('epa_tree')
    # display(SVG(my_tree.pipe(format = 'svg')))

    saved_forest = filename
    joblib.dump(rf_model, saved_forest)

def model_test(X_test, y_test, filename):
    rf_model = joblib.load(filename)
    rf_y_pred = rf_model.predict(X_test)

    matrix = pd.DataFrame(confusion_matrix(y_test, rf_y_pred), columns = ['Low', 'High'], index = ['Low', 'High'])
    fig_path = os.path.join(os.getcwd(), "measures.pdf")
    pdf = matplotlib.backends.backend_pdf.PdfPages(fig_path)
    fig = plt.figure()
    sns.heatmap(matrix, annot = True, cbar = True, cmap = 'Blues', fmt = 'g')
    plt.ylabel('True Match Quality')
    plt.xlabel('Predicted Match Quality')
    pdf.savefig(fig)
    plt.close()
    print(matrix)

    probabilities = rf_model.predict_proba(X_test)
    probabilities = probabilities[:, 1]
    score = roc_auc_score(y_test, probabilities)
    fig = plt.figure()
    fp, tp, _ = roc_curve(y_test, probabilities, pos_label = 2)
    plt.plot(fp, tp, marker = '.', label = 'ROC (area = {:.3f})'.format(score))
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc = 'best')
    pdf.savefig(fig)
    plt.close()

    print('Accuracy:', accuracy_score(y_test, rf_y_pred)*100)

    pdf.close()

    return rf_y_pred

def model_predict(X, filename):
    rf_model = joblib.load(filename)
    rf_y_pred = rf_model.predict(X)
    return rf_y_pred