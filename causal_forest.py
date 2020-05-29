import numpy as np
import statsmodels.formula.api as sm
import statsmodels.stats.api as sms
import pandas as pd
import random
import matplotlib.pyplot as plt
import scipy.interpolate

from tqdm import tqdm
from sklearn import tree
import time
from functools import partial, reduce
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, ExtraTreeClassifier

from sklearn.tree._tree import TREE_LEAF

# true implementation of the honest tree; takes much longer
try:
    from decision_tree import Tree
except:
    pass

class CausalForest(object):
    """
    Causal Forest: The object
    arg:
        use_w_in_tree: parameter that controls the use of the w variable when deciding the cuts,
        true_honest_tree: will we use the true honest tree implementation? if yes, takes much longer and not much difference
            (only used when algorithm = 'double_sample')
    """
    
    def __init__(self, use_w_in_tree=False, true_honest_tree=False):
        self.use_w_in_tree = use_w_in_tree
        self.true_honest_tree = true_honest_tree
        
    def _prune_tree(self, model, I, W_I, w_var, min_samples_leaf):
        """
        function that prunes the tree until there are min_samples_leaf samples from eachtreatment class 
        (from the I sample); one of the requirements of the honest tree
        """
        
        # let's build a dataframe with the information of how many samples from each treatm. class are in each leaf
        index_cols = list(I[[]].index.names)
        preds_W = I[[]].reset_index().merge(W_I.reset_index(), on=index_cols, how='left')
        
        
        def prune_index(tree, index, leaves_to_remove):
            '''
            function that removes a leaf and makes its nodes point to the original's parents
            '''
            
            def get_index(x, children):
                if x in children:
                    return list(children).index(x)
                else:
                    return None

            # parents of the leaves we are about to remove
            parents = ([get_index(x, tree.children_right) for x in leaves_to_remove] + 
                       [get_index(x, tree.children_left) for x in leaves_to_remove])
            
            # filtering the empty ones
            parents = [x for x in parents if x]
            parents_set = list(set(parents))
            
            for parent in parents_set:
                tree.children_left[parent] = TREE_LEAF
                tree.children_right[parent] = TREE_LEAF
                
        def get_leaves_to_remove(I, preds_W, model, w_var, min_samples_leaf, first_children):
            """
            function that calculates which leaves must be pruned
            """
            preds_W['leaf'] = model.apply(I)
            
            # how many samples for each treatm. class
            leaves_to_remove = preds_W.groupby(['leaf'] + w_var).size().reset_index()
            
            # pivoting so that we capture the leaves with no samples of one of the classes
            leaves_to_remove = leaves_to_remove.pivot_table(index='leaf', columns=w_var, values=0).fillna(0).reset_index()
            
            # melting to its original form
            leaves_to_remove = leaves_to_remove.melt(id_vars='leaf', value_name='leaf_size')
            
            # finding the incomplete nodes
            leaves_to_remove = leaves_to_remove.query("leaf_size < @min_samples_leaf").leaf.unique().tolist()
            
            # removing the parent nodes
            leaves_to_remove = [x for x in leaves_to_remove if x not in first_children]
            
            return preds_W, leaves_to_remove
        
        
        # getting the first noees - we can't remove them in case they don't satisfy our conditions, the program loops
        first_children = [model.tree_.children_right[0], model.tree_.children_left[0], 0]
        
        preds_W, leaves_to_remove = get_leaves_to_remove(I, preds_W, model, w_var, min_samples_leaf, first_children)
        
        # looping until ready
        while leaves_to_remove:
            # start from the root
            prune_index(model.tree_, 0, leaves_to_remove)
            
            # updating the nodes
            preds_W, leaves_to_remove = get_leaves_to_remove(I, preds_W, model, w_var, min_samples_leaf, first_children)
            
        return model
    
    def _train_honest_tree(self, df, y_var, w_var, index_cols, min_samples_leaf):
        """
        function that effectively trains each tree in the forest
        """
        
        if self.algorithm == 'double_sample':
            # step 0 : subsample of df to populate I and J
            df_sample, df_not_sample = train_test_split(df, test_size=0.2)
            
            df_out = df_not_sample.set_index(index_cols).drop(y_var + w_var*(not self.use_w_in_tree), 1)
            y_out = df_not_sample.set_index(index_cols)[y_var]
            W_out = df_not_sample.set_index(index_cols)[w_var]
            
            s = 0.5
        elif self.algorithm == 'propensity':
            df_sample = df
            s = np.random.uniform(0.3, 0.5)
            
        # step 1 : splitting (J = train, I = predictions)
        J, I, tau_J, tau_I, W_J, W_I = train_test_split(
                                df_sample.set_index(index_cols).drop(y_var + w_var*(not self.use_w_in_tree), 1),
                                df_sample.set_index(index_cols)[y_var],
                                df_sample.set_index(index_cols)[w_var],
                                test_size=s)
        
        # step 2 : training the tree
        if self.algorithm == 'double_sample':
            if not self.true_honest_tree:
                model = DecisionTreeRegressor(criterion='mse', min_samples_leaf=2*min_samples_leaf)
            else:
                model = decision_tree.DecisionTree(min_samples_leaf)
                
            # J is used for training
            model.fit(J, tau_J)
            
            # I is used for prediction and pruning
            model = self._prune_tree(model, I, W_I, w_var, min_samples_leaf)
            X_prediction, tau_prediction, W_prediction = I, tau_I, W_I
            
        elif self.algorithm == 'propensity':
            model = ExtraTreeClassifier(criterion='gini', min_samples_leaf=2*min_samples_leaf, splitter='random')
            
            # we use J for training, but this time the target is the treament class variable
            model.fit(J, W_J)
            
            # pruning and prediction in J
            model = self._prune_tree(model, I, W_I, w_var, min_samples_leaf)
            X_prediction, tau_prediction, W_prediction = J, tau_J, W_J
            
        
        # creating a dataframe with the predictions by leaf
        leaves = X_prediction[[]].copy()
        leaves['leaf'] = model.apply(X_prediction)
        leaves['true'] = tau_prediction
        leaves[w_var] = W_prediction
        leaves = leaves.groupby(['leaf']+w_var).true.mean().reset_index()
        
        # predicting
        if self.full_predictor and self.algorithm == 'double_sample':
            # if full, we predict for everyone
            X_prediction = pd.concat([df_out, X_prediction])
        
        this_preds = X_prediction[[]].copy()
        this_preds['leaf'] = model.apply(X_prediction)
        this_preds.reset_index(inplace=True)
        
        return leaves, this_preds, model
    
    def _adjust_model(self, model, leaves, tree_index):
        """
        function that replaces the fitted model prediction values to those of difference between classes
        """
        if self.algorithm == 'propensity':
            # classifier cant predict non binary labels, so we do this workaround in this case
            self.propensity_score[tree_index] = {int(x['leaf']): x['difference'] for index, x in leaves.iterrows()}
            return model
        for indes, row in leaves.iterrows():
            print(row['difference'])
            model.tree_.value[int(row['leaf'])] = row['difference']
        return model
            
    def _get_tree_result(self, tree_index, df, y_var, w_var, index_cols, min_samples_leaf):
        leaves, preds, model = self._train_honest_tree(df, y_var, w_var, index_cols, min_samples_leaf)
        
        # pivoting
        leaves = leaves.pivot_table(values='true', index='leaf', columns=w_var)
        
        # changing the names
        leaves.columns = [w_var[0] + '_' + str(x) for x in leaves.columns]

        # claculating the difference
        leaves['difference'] = leaves[w_var[0] + '_1'] - leaves[w_var[0] + '_0']
        
        # merging predictions with leaves
        effects = preds.merge(leaves.reset_index(), on='leaf', how='left')
        
        # adjusting the model
        model = self._adjust_model(model, leaves.reset_index(), tree_index)
        
        return effects.set_index(index_cols)[['difference']].rename(columns={'difference':f'diff_tree_{tree_index}'}), model
    
    def fit(self, df, y_var, w_var, index_cols, min_samples_leaf, n_estimators, algorithm, n_threads=20, full_predictor=True):
        if algorithm not in ['double_sample', 'propensity']:
            raise Exception(NotImplemented)
            
        if algorithm == 'propensity':
            self.propensity_score = [None for _ in range(n_estimators)]
        
        if not isinstance(y_var, list):
            y_var = [y_var]
            
        if not isinstance(w_var, list):
            w_var = [w_var]
            
        if index_cols is None or not index_cols:
            assert 'index' not in df.columns
            df.reset_index(inplace=True)
            index_cols = ['index']
            
        new_index_cols = index_cols.copy()
        
        # asserting index_cols are unique
        if df[index_cols].duplicated().sum() > 0:
            df.reset_index(drop=True, inplace=True)
            index_name = 'new_index'
            df.reset_index(drop=False, inplace=True)
            new_index_cols += [index_name]
            
            
        # sabvin for later use
        self.w_var = w_var
        self.index_cols = new_index_cols
        self.columns = list(df.columns)
        self.y_var = y_var
        self.algorithm = algorithm
        self.full_predictor = full_predictor
        
        df[w_var[0]] = df[w_var[0]].astype(int)
        partial_func = partial(self._get_tree_result, df=df, y_var=y_var, w_var=w_var, index_cols=new_index_cols, 
                               min_samples_leaf=min_samples_leaf)
        
        if n_threads > 1:
            p = ThreadPool(n_threads)
            res = p.map(partial_func, range(n_estimators))
            p.close()
            p.join()
        else:
            res = []
            for i in range(n_estimators):
                res.append(partial_func(i))
                
        # aggregating the results
        # res is a list of tuples
        model = [x[1] for x in res]
        
        final = reduce(lambda left, right : pd.merge(left, right, on=new_index_cols, how='outer'),
                      [x[0].reset_index() for x in res]).set_index(new_index_cols)
        
        self.model = model
        
        return final
    
    def predict(self, df):
        """
        function that predicts the expected effect
        """
        if not hasattr(self, 'model'):
            raise Exception('Model not fitted.')
            
        # removing 'index' from index_cols if necessary
        index_cols = self.index_cols.copy()
        if 'new_index' in self.index_cols:
            index_cols.remove('new_index')
            
            
        # removing w from df, if necessary
        if not self.use_w_in_tree:
            for col in self.w_var:
                if col in df.columns:
                    index_cols += [col]
        
        # removing y from df, if necessary
        for col in self.y_var:
            if col in df.columns:
                index_cols += [col]

        preds = df.set_index(index_cols)[[]].copy()
        for index, tree in enumerate(self.model):
            if self.algorithm == 'propensity':
                preds[f'pred_tree_{index}'] = (pd.Series(tree.apply(df.set_index(index_cols))).map(
                                            self.propensity_score[index]).tolist())
            else:
                preds[f'pred_tree_{index}'] = tree.predict(df.set_index(index_cols))
        preds['prediction'] = preds.mean(axis=1)
        
        return preds
    
    @property
    def feature_importances_(self):
        """
        self explanatory
        """
        
        if not hasattr(self, 'model'):
            raise Exception('Model not fitted.')
            
        # removing 'index' from index_cols if necessary
        index_cols = self.index_cols.copy()
        if 'new_index' in self.index_cols:
            index_cols.remove('new_index')
            
            
        # removing w from df, if necessary
        if not self.use_w_in_tree:
            for col in self.w_var:
                if col in self.columns:
                    index_cols += [col]
                    
        final_cols = [x for x in self.columns if x not in index_cols and x not in self.y_var]
    
        imp = pd.DataFrame([x.feature_importances_.tolist() for x in self.model]).T
        imp = imp.mean(axis=1).to_frame()
        imp.columns = ['importance']
        imp['var'] = final_cols
        
        return imp.sort_values('importance', ascending=False)
    
    
    def plot_results(self, df):
        """
        function that plots the calculated effects vs the real effects in the population;
        ideal curve must be monotonic
        """
        res = df.copy()
        def compute_mean_diff(c):
            """
            computes the confidence interval between two series
            """
            cm = sms.CompareMeans(sms.DescrStatsW(c.query("{} == 1".format(self.w_var[0]))[self.y_var[0]]),
                                  sms.DescrStatsW(c.query("{} == 0".format(self.w_var[0]))[self.y_var[0]])
                                 )
            return cm.tconfint_diff(usevar='unequal')
        
        def aggregator(x):
            # the bin number
            score_group = x['bin'].cat.codes.unique()[0]
            
            # score given by the model
            score = x['prediction'].mean()
            
            # thresholds
            up, down = compute_mean_diff(x)
            
            # class proportion
            pr_treatment = x[self.w_var[0]].mean()
            
            # difference between means
            tau = (x.query("{} == 1".format(self.w_var[0]))[self.y_var[0]].mean() - 
                   x.query("{} == 0".format(self.w_var[0]))[self.y_var[0]].mean() )
            
            res = pd.DataFrame(data=[[score_group, score, up, down, pr_treatment, tau]], 
                              columns=['score_group', 'score', 'upper', 'lower', 'pr_treatment', 'tau'])
            return res
            
        res['prediction'] = self.predict(res)['prediction'].tolist()
        res['bin'] = pd.qcut(res['prediction'], 20)
        resp = res.groupby('bin').apply(aggregator)
        resp['lift'] = resp['tau'] / resp['tau'].mean()
        
        bp_list = [(row['lower']*100, row['tau'], row['upper']) for _, row in resp.iterrows()]
        
        fig = plt.figure(figsize=(10, 8))
        plt.boxplot(bp_list)
        
        plt.title('Model evaluation')
#         plt.ylim(-120, )
        plt.xlabel('Score group (model prediction)')
        plt.ylabel('Treatment effect (real)')
        plt.grid()
        plt.show()
        
    def plot_histogram(self, df):
        """
        function that plots a simple histogram for the effects
        """
        preds = self.predict(df)
        plt.hist(preds)
        plt.show()
