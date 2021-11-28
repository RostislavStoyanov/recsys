import pandas as pd

#based on https://maxhalford.github.io/blog/target-encoding/
class TargetEncoder():
    
    def __init__(self, cols, w):
        
        if isinstance(cols, str):
            self.cols = [cols]
        else: self.cols = cols

        self.w = w
        
    def fit(self, X, y):
        #calculate target overall mean
        if y not in X.columns:
             raise ValueError('Column: {} not in dataframe'.format(y))
        
        self.target = y
        self.target_mean= X[self.target].mean()
        
        #store mapping for each column w.r.t target column
        self.maps = {}
        for col in self.cols:
            
            if col not in X:
                self.maps = {} # valid state if we decide to transform nothing happens
                raise ValueError('Column: {} not in dataframe'.format(col))

            aggr = X.groupby(col)[self.target].agg(['count', 'mean'])
            counts = aggr['count']
            means = aggr['mean']

            self.maps[col] = (counts * means + self.w * self.target_mean) / (counts + self.w)

        return self

    
    def transform(self, X, y=None):
        X_new = X.copy()
        for col, smooth in self.maps.items():
            new_col_name = col + '_' + self.target + "_TE"
            X_new[new_col_name] = X_new[col].map(smooth)
            X_new[new_col_name].fillna(self.target_mean, inplace=True)
        return X_new
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)