from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve, accuracy_score, recall_score, \
    precision_score, f1_score, roc_auc_score, precision_recall_fscore_support
from sklearn import naive_bayes, svm
import matplotlib.pyplot as plt


#--------------------------------------------------------------------------------------------------------------------------------------------
def scores(X_test,
           predictions,
           clf,
           Y_test):

    y_scores_df = clf.decision_function(X_test)
    y_scores_df = (y_scores_df - y_scores_df.min()) / (y_scores_df.max() - y_scores_df.min())

    # Métricas
    print('___________________________________________________________')
    cm1 = confusion_matrix(Y_test, predictions)

    print('Confusion Matrix:')
    print(cm1)
    print(str(cm1[1, 1]) + "\t" + str(cm1[1, 0]))
    print(str(cm1[0, 1]) + "\t" + str(cm1[0, 0]))
    print("--------")

    prfs = precision_recall_fscore_support(Y_test, predictions)
    print('precision_recall_fscore_support: ')
    print(prfs)
    print('----')

    # Calculo
    sensitivity1 = cm1[1, 1] / (cm1[1, 1] + cm1[1, 0])
    specificity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    precision1 = cm1[1, 1] / (cm1[1, 1] + cm1[0, 1])
    precision2 = cm1[0, 0] / (cm1[0, 0] + cm1[1, 0])
    f11 = (2 * precision1 * sensitivity1) / (precision1 + sensitivity1)
    f12 = (2 * precision2 * specificity1) / (precision2 + specificity1)


    acc = str(accuracy_score(Y_test, predictions))
    print("recall: " + str(recall_score(Y_test, predictions)))
    print("precision: " + str(precision_score(Y_test, predictions)))
    print("precision1: " + str(precision1))
    print("precision2: " + str(precision2))
    print('Sensitivity : ', sensitivity1)
    print('Specificity : ', specificity1)
    print("f1: " + str(f1_score(Y_test, predictions)))
    print("f11: " + str(f11))
    print("f12: " + str(f12))
    print("accuracy: " + str(acc))

    return y_scores_df, prfs, cm1, acc

#--------------------------------------------------------------------------------------------------------------------------------------------
def scores_Proba(X_test,
                 predictions,
                 clf,
                 Y_test):
    y_scores_df = clf.predict_proba(X_test)
    y_scores_df = y_scores_df[:, 1]


    # Métricas
    print('___________________________________________________________')
    cm1 = confusion_matrix(Y_test, predictions)

    print('Confusion Matrix:')
    print(cm1)
    print(str(cm1[1, 1]) + "\t" + str(cm1[1, 0]))
    print(str(cm1[0, 1]) + "\t" + str(cm1[0, 0]))
    print("--------")

    prfs = precision_recall_fscore_support(Y_test, predictions)
    print('precision_recall_fscore_support: ')
    print(prfs)
    print('----')

    # Calculo
    sensitivity1 = cm1[1, 1] / (cm1[1, 1] + cm1[1, 0])
    specificity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    precision1 = cm1[1, 1] / (cm1[1, 1] + cm1[0, 1])
    precision2 = cm1[0, 0] / (cm1[0, 0] + cm1[1, 0])
    f11 = (2 * precision1 * sensitivity1) / (precision1 + sensitivity1)
    f12 = (2 * precision2 * specificity1) / (precision2 + specificity1)



    acc = str(accuracy_score(Y_test, predictions))
    print("recall: " + str(recall_score(Y_test, predictions)))
    print("precision: " + str(precision_score(Y_test, predictions)))
    print("precision1: " + str(precision1))
    print("precision2: " + str(precision2))
    print('Sensitivity : ', sensitivity1)
    print('Specificity : ', specificity1)
    print("f1: " + str(f1_score(Y_test, predictions)))
    print("f11: " + str(f11))
    print("f12: " + str(f12))
    print("accuracy: " + str(acc))

    #print("roc_auc: " + str(roc_auc_score(Y_test, y_scores_df)))


    return y_scores_df, prfs, cm1, acc



#--------------------------------------------------------------------------------------------------------------------------------------------
def plot_auc(y_test,
             y_score,
             estimator_name,
             filename):

    false_positive_r, true_positive_r, thresholds = roc_curve(y_test, y_score, pos_label=1)
    roc_auc = auc(false_positive_r, true_positive_r)
    print("estimator name:" + estimator_name)
    print("roc_curve(y_test, y_pred): FP:" + str(false_positive_r) + "     TP:" + str(true_positive_r))
    print("auc(FP, TP): " + str(roc_auc))


    label = estimator_name + ' classifier (Area = {:.1f}%)'.format(roc_auc * 100)

    plt.plot([0, 1], [0, 1], linestyle='--', label='No skill classifier')  #, 'k--')
    plt.plot(false_positive_r, true_positive_r, label=label)

    plt.title('ROC score(s)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', prop={'size': 8})

    plt.grid()
    plt.show()
    return roc_auc


#--------------------------------------------------------------------------------------------------------------------------------------------
#average the word vectors for a set of words
def make_feature_vec (words, model, num_features) :
    feature_vec = np.zeros((num_features,), dtype= 'float')
    nwords = 0
    index2word_set = set(model.wv.index2word)

    for word in words :
        if word in index2word_set:
            nwords = nwords +1
            feature_vec = np.add(feature_vec, model[word])
    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec


#calculate average feature vectors for all poems
def get_avg_feature_vecs(poems, model, num_features) :
    counter = 0
    poem_feature_vecs = np.zeros((len(poems), num_features), dtype='float')

    for poem in poems:
        poem_feature_vecs[counter] = make_feature_vec(poem, model, num_features)
        counter = counter + 1
    return poem_feature_vecs


# convert a poem to a list of words.

def poem_to_wordlist(poem):
    # convert to lower case and split at whitespace
    words = poem.lower().split()
    return words



#the dataset is loaded
poems = pd.read_csv(r'C:\Users\Mimi\Desktop\foralgorithms2.csv')



# the different models are loaded
model1 = Word2Vec.load(r'C:\Users\Mimi\Desktop/fourth cbow') # word2vec with cbow , no transfer learning
model2 = Word2Vec.load(r'C:\Users\Mimi\Desktop/fourth skipgram') # word2vec with skipgram , no transfer learning
model3 = Word2Vec.load(r'C:\Users\Mimi\Desktop/TLfourthcbow') # word2vec with cbow and transfer learning
model4 = Word2Vec.load(r'C:\Users\Mimi\Desktop/TLfourthskipgram') # word2vec with skipgram and transfer learning

model1.wv.syn0.shape
model2.wv.syn0.shape
model3.wv.syn0.shape
model4.wv.syn0.shape


model = model4
num_features = 300
X = poems['cleanPoem']
y = poems['poemClass']
skf = StratifiedKFold(n_splits=10)




#the classifiers
forest = RandomForestClassifier(n_estimators = 100)
dtc = DecisionTreeClassifier()
naiveG = naive_bayes.GaussianNB()
SVM = svm.SVC(kernel='rbf')




# Iterate over each train-test split
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    # Split train-test
    X_train, X_test = poems.iloc[train_index], poems.iloc[test_index]
    y_train, y_test = poems.iloc[train_index], poems.iloc[test_index]

    clean_train_poems = []
    for poem in X_train['cleanPoem']:
        clean_train_poems.append(poem_to_wordlist(poem))
    trainDataVecs = get_avg_feature_vecs(clean_train_poems, model, num_features)

    clean_test_poems = []
    for poem in X_test['cleanPoem']:
        clean_test_poems.append(poem_to_wordlist(poem))
    testDataVecs = get_avg_feature_vecs(clean_test_poems, model, num_features)

    # Train the model
    forest = forest.fit(trainDataVecs, y_train['poemClass'])

    print("Predicting labels for test data..")
    result_forest = forest.predict(testDataVecs)

    # Metrics
    y = np.array(y_test.loc[:, y_test.columns == 'poemClass'])
    # -------------------
    print('****************************************** Random Forest ******************************************')

    [y_scores_df, prfs, cm1, acc] = scores_Proba(testDataVecs,
                                           result_forest,
                                           forest,
                                           y)

    plt.figure()
    roc_auc = plot_auc(y,
                       y_scores_df,
                       'Forest',
                       "result_forest" + ' RO ROC' + str(
                           1) + ".png")  # Example 13 #https://www.programcreek.com/python/example/81207/sklearn.metrics.roc_curve


























