######################################################################
# Main class that contains every classification on dense and sparse VSMs
# Models Evaluation(F1) and Missclassified Test Samples
#####################################################################
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import pickle,glob
from sklearn import svm
from sklearn.svm import SVC,LinearSVC
from tqdm import tqdm, trange
import numpy as np
import sys
from sklearn.pipeline import make_pipeline
from emb_vector import TFTransformer
import pandas

class Sentiment_Model:
    def classify_batch(self, x_train, y_train, x_test, y_test, info, transformer, models=None):
        if models:
            status_bar = trange(len(models), desc='', leave=True)
            for model in status_bar:
                status_bar.set_description("Training Model [{}]".format(models[model]), refresh=True)
                # status_bar.refresh() # to show immediately the update
                classifier = getattr(self, models[model])
                classifier(x_train, y_train, x_test, y_test, info, transformer)

    def ensemble(self, x_train, y_train, x_test, y_test, info ):
        from mlxtend.classifier import EnsembleVoteClassifier

        ensemble_models = list()
        for folder_model in glob.iglob('./pickled/models/*', recursive=True):
            model_ = folder_model.rsplit('\\')[1]
            if model_ == 'ensemble':
                continue

            for file_model in glob.iglob('./pickled/models/{}/*'.format(model_)):
                _model = file_model.split('\\')[1].strip('.pkl')
                if _model == '{0}_{1}'.format(info['embedding'], info['dataset']):
                    print(folder_model)
                    ensemble_models.append(self.load_model('{0}/{1}'.format(model_, _model))[0].best_estimator_)

        ensembler_ = EnsembleVoteClassifier(clfs=ensemble_models, weights=[1,1], voting='hard')
        ensembler_.fit(x_train, y_train)

        y_predicted_ = ensembler_.predict(x_test)
        print('Voting Classifier Prediction: ________')
        print(classification_report(y_test, y_predicted_))
        self.save_model(ensembler_, name='ensembled/mlxtend_imdb')


    def nb(self, x_train, y_train, x_test, y_test, info, transformer=None):
        param_grid={'alpha': [1.0]}

        if transformer and not transformer.get_feature_names():
            param_grid = {
                          'tftransformer__min_df': [2],
                          'tftransformer__ngram_range': [(1,1), (1,2), (1,3)]
                         }
            estimator_ = make_pipeline(transformer, MultinomialNB())
        # We are dealing with embedding vector HERE
        elif not transformer:
            estimator_ = make_pipeline(transformer, MultinomialNB())
        # We are dealing with pre-fitted TF_Vectorizer
        else:
            estimator_ = MultinomialNB()

        nb_cv=GridSearchCV(estimator=estimator_,param_grid=param_grid,cv=5, n_jobs=1)
        nb_cv.fit(x_train,y_train)

        y_predicted_ = nb_cv.predict(x_test)
        final_obj = [nb_cv, classification_report(y_test, y_predicted_)]

        name = 'nb/{0}_{1}'.format(info.get('embedding'), info.get('dataset'))
        self.save_model(final_obj, name)


    def lr(self, x_train, y_train, x_test, y_test, info, transformer=None):
        param_grid={"C":np.logspace(-3,3,7)}

        if isinstance(transformer, TFTransformer):
            if not transformer.get_feature_names():
                param_grid = {
                              'tftransformer__min_df': [2],
                              'tftransformer__ngram_range': [(1,1), (1,2), (1,3)],
                              'logisticregression__C': [0.01, 0.1, 0.25, 1.0],
                              'logisticregression__penalty': ['l1','l2']
                            }
                estimator_ = make_pipeline(transformer, LogisticRegression(solver='saga'), max_iter=10000)
        else:
            estimator_ = LogisticRegression(solver='saga', max_iter=10000)

        logreg_cv=GridSearchCV(estimator=estimator_,param_grid=param_grid,cv=5, n_jobs=4)
        logreg_cv.fit(x_train,y_train)

        y_predicted_ = logreg_cv.predict(x_test)
        final_obj = [logreg_cv, classification_report(y_test, y_predicted_)]

        name = 'lr/{0}_{1}'.format(info.get('embedding'), info.get('dataset'))
        self.save_model(final_obj, name)

    def svm(self, x_train, y_train, x_test, y_test, info, transformer=None):
        # parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':(1,0.25,0.5,0.75),'gamma': (1,2,3,'auto'), 'degree': (2,3)}
        # param_grid={"C": np.logspace(-3,3,7)}# l1 lasso l2 ridge
        param_grid={"C": [0.01, 0.1, 1.0]}# l1 lasso l2 ridge

        if isinstance(transformer, TFTransformer):
            if not transformer.get_feature_names():
                param_grid = {
                              'tftransformer__min_df': [2],
                              'tftransformer__ngram_range': [(1,1), (1,2), (1,3)],
                              'linearsvc__C': [0.01, 0.1, 1.0]
                            }
                estimator_ = make_pipeline(transformer, LinearSVC(max_iter=10000))
        # We are dealing with embedding vector HERE
        else :
            estimator_ = LinearSVC(max_iter=10000)

        svm_cv=GridSearchCV(estimator=estimator_, param_grid=param_grid, cv=5, n_jobs=4)
        svm_cv.fit(x_train, y_train)

        y_predicted_ = svm_cv.predict(x_test)
        final_obj = [svm_cv, classification_report(y_test, y_predicted_)]
        name = 'svm/{0}_{1}'.format(info.get('embedding'), info.get('dataset'))

        self.save_model(final_obj, name)

    def ridge(self, x_train, y_train, x_test, y_test, info):
        # grid = {'kernel':('linear', 'rbf', 'poly')}
        # clf = GridSearchCV(SVC(gamma="scale"), parameters, cv=5, n_jobs=8)
        # grid = {'solver':['auto']}
        RC = RidgeClassifier()
        RC.fit(x_train, y_train)

        y_predicted_ = RC.predict(x_test)
        final_obj = [RC, classification_report(y_test, y_predicted_)]
        name = 'ridge/{0}_{1}'.format(info.get('embedding'), info.get('dataset'))

        self.save_model(final_obj, name)

    def rf(self, x_train, y_train, x_test, y_test, info, transformer=None):

        param_grid={'n_estimators':[400,800], 'max_depth':[12,24]}
        # If transformer is on for train/test
        if isinstance(transformer, TFTransformer):
            if not transformer.get_feature_names():
                param_grid = {
                              'tftransformer__ngram_range': [(1,1), (1,2), (1,3)],
                              'randomforestclassifier__n_estimators': [400, 800],
                              'randomforestclassifier__max_depth': [12,24],
                             }
                estimator_ = make_pipeline(transformer, RandomForestClassifier())
        # We are dealing with embedding vector HERE
        else:
            estimator_ = RandomForestClassifier()

        rf_cv=GridSearchCV(estimator=estimator_,param_grid=param_grid,cv=5, n_jobs=4)
        rf_cv.fit(x_train,  y_train)

        # See what kind of pipe it is
        #PASSSED
        if 'randomforestclassifier__max_depth' in rf_cv.best_params_:
            adaboost = AdaBoostClassifier(base_estimator=RandomForestClassifier(
            n_estimators=rf_cv.best_params_['randomforestclassifier__n_estimators'],
            max_depth=rf_cv.best_params_['randomforestclassifier__max_depth']))
            transformer.set_params(ngram_range=rf_cv.best_params_['tftransformer__ngram_range'])
            estimator_ = make_pipeline(transformer, adaboost)
        # TODO:: TESTING
        else:
            estimator_ = AdaBoostClassifier(base_estimator=RandomForestClassifier(
            n_estimators=rf_cv.best_params_['n_estimators'],
            max_depth=rf_cv.best_params_['max_depth']))

        estimator_.fit(x_train, y_train)

        y_predicted_ = estimator_.predict(x_test)
        final_obj = ['dummy', classification_report(y_test, y_predicted_)]

        name = 'rf/{0}_{1}'.format(info.get('embedding'), info.get('dataset'))
        self.save_model(final_obj, name)

    def overview(self, model):
        print("Best parameters set found on development set:")
        print()
        print(model.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = model.cv_results_['mean_test_score']
        stds = model.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, model.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

    def missclassified(self, y_gold, y_pred, y_text):
        pd = pandas.DataFrame(columns=['gold', 'predicted', 'text'])
        pd['gold'] = y_gold
        pd['predicted'] = y_pred
        pd['text'] = y_text
        pd = pd.query('gold != predicted')
        return pd


    def save_model(self, obj, name ):
        with open('./pickled/models/{}.pkl'.format(name), 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, name ):
        with open('./pickled/models/{}.pkl'.format(name), 'rb') as f:
            return pickle.load(f)
