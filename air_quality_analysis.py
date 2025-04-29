import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
df = pd.read_csv("../input/datasetucimlairquality/AirQualityUCI.csv", parse_dates={'datetime': ['Date', 'Time']})
print(df.head())
print(df.info(), df.shape)
print(df.describe(), df['CO_level'].value_counts())
columns = {"Date", "Time", "CO_GT", "PT08_S1_CO", "NMHC_GT", "C6H6_GT", "PT08_S2_NMHC", "Nox_GT", "PT08_S3_Nox", "NO2_GT", "PT08_S4_NO2", "PT08_S5_O3", "T", "RH", "AH", "CO_level"}
def findNumOfMissingVals(col):
    allValues = []    
    for column in col:
        val = 0 
        for i in df[column]:
            if i == -200:
                val += 1
        allValues.append(val)                         
    return allValues
values = findNumOfMissingVals(['AH','C6H6_GT','CO_GT','CO_level','datetime','NMHC_GT','NO2_GT','Nox_GT','PT08_S1_CO','PT08_S2_NMHC','PT08_S3_Nox','PT08_S4_NO2','PT08_S5_O3','RH','T'])
valuesMapped = dict(zip(sorted(columns), values))
print(valuesMapped)
df.drop('NMHC_GT', axis=1, inplace=True)
def replaceValues(value1, value2, df):
    df = df.replace(value1, value2)
    return df
df2 = df.copy()
df2 = replaceValues(-200, np.nan, df2)
def dropMissingRows(df):
    df = df.dropna(axis=0)
    return df
df2 = dropMissingRows(df2)
df2 = df2.sort_values(by=['datetime'], ascending=True)
print(df2.head())
print(df2.shape)
def visualizeScatter(df, x, y, fill, title):
    fig = px.scatter(df, 
    x=x, 
    y=y, 
    color=fill, 
    title=title)
    
    fig.show()
def visualizeFilledArea(x, y, fill, hd):
    fig = px.area(df, 
    x=x, 
    y=y,
    color=fill,
    hover_data=[hd])
  
    fig.show()
def visualizeLineCharts(df, columns, x1, title):   
    columns = columns 
    
    fig = go.Figure([{
    'x': df[x1],
    'y': df[col],
    'name': col
    }  for col in columns], layout=go.Layout(title=go.layout.Title(text=title)))
    
    fig.show()
def showViolinPlot(x, y, x2, y2, x3, y3):
    fig, axes = plt.subplots(1,3, figsize=(25, 5))
    sns.violinplot(x=x, y=y, data=df2, hue=x, palette='rocket', ax=axes[0])
    axes[0].set_title("{} by {}".format(x, y))

    
    sns.violinplot(x=x2, y=y2, data=df2, hue=x2, palette='rocket', ax=axes[1])
    axes[1].set_title("{} by {}".format(x2, y2))

    
    sns.violinplot(x=x3, y=y3, data=df2, hue=x3, palette='rocket', ax=axes[2])
    axes[2].set_title("{} by {}".format(x3, y3))
    
def displayHeatMap(dim1, dim2, title, df):
    fig=plt.figure(figsize=(dim1,dim2))
    plt.title(title)
    sns.heatmap(df, annot= True, cmap='flare')
def showDistributions(category1, category2, category3):
    fig, axes = plt.subplots(1,3, figsize=(25, 5))
    sns.histplot(data=df2, x=category1, kde=True, color="darkseagreen", ax=axes[0])
    axes[0].set_title("Distribution of {}".format(category1))
    sns.histplot(data=df2, x=category2, kde=True, color="darkseagreen", ax=axes[1])
    axes[1].set_title("Distribution of {}".format(category2))
    sns.histplot(data=df2, x=category3, kde=True, color="darkseagreen", ax=axes[2])
    axes[2].set_title("Distribution of {}".format(category3))
showDistributions("PT08_S1_CO", "PT08_S2_NMHC", "Nox_GT")
showDistributions("CO_GT", "C6H6_GT", "PT08_S3_Nox")
showDistributions("NO2_GT", "PT08_S4_NO2", "PT08_S5_O3")
corr = df.corr()
displayHeatMap(12, 12, 'Correlation Between Air Quality Attributes', corr)
twentyFourHrSpanDf = df2[6:28].copy()
visualizeLineCharts(twentyFourHrSpanDf, ['CO_GT', 'C6H6_GT', 'Nox_GT', 'NO2_GT'], 'datetime', "Air Pollutant Concentrations Change Over 2004-03-11")
firstDaydf = df2[:6].copy()
visualizeLineCharts(firstDaydf, ['CO_GT', 'C6H6_GT', 'Nox_GT', 'NO2_GT'], 'datetime', "Air Pollutant Concentrations Change Over 2004-03-10")
lastDaydf = df[9343: -8].copy()
lastDaydf.drop('CO_GT', axis=1, inplace=True)
visualizeLineCharts(lastDaydf, ['C6H6_GT', 'Nox_GT', 'NO2_GT'], 'datetime', "Air Pollutant Concentrations change over 2005-03-13")
dfWeekday = df['datetime'].dt.day_name()
df['week_day'] = dfWeekday
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',  'Friday',   'Saturday',  'Sunday']
def createSub(lst, col):
    for i in df.index:
        test_days = list(df[col][i:i+7])
        if test_days == lst:
            week_df = df.iloc[i:i+7,:]
            break
    return week_df
week_df = createSub(days, 'week_day')    
visualizeLineCharts(week_df, ['C6H6_GT', 'Nox_GT', 'NO2_GT'], 'datetime', "Air Pollutant Concentrations change over 1 week")
visualizeScatter(df2, 'RH', 'T', 'RH', 'Concentrations over Temperature and Relative Humidity')
visualizeScatter(df2, 'AH', 'T', 'AH', 'Concentrations over Temperature and Absolute Humidity')
visualizeScatter(df2, 'CO_level', 'datetime', 'CO_level', 'Concentrations of CO_level over the Year')
def createDataSubset(df, range1, range2):
    first = df[range1:range2]
    return first
visualizeScatter(createDataSubset(df2, 0, 3470), 'datetime', 'Nox_GT', 'Nox_GT', 'Nox_GT Concentrations over First half of year')
visualizeScatter(createDataSubset(df2, 3470, 6941), 'datetime', 'Nox_GT', 'Nox_GT', 'Nox_GT Concentrations over second half of year')
showViolinPlot('CO_level', 'T', 'CO_level', 'CO_GT', 'CO_level', 'RH')
visualizeLineCharts(df2, ["CO_GT", "PT08_S1_CO", "C6H6_GT"], 'datetime', "1 Year Time Span of Air Concentrations")
visualizeLineCharts(df2, ["Nox_GT", "PT08_S3_Nox", "NO2_GT"], 'datetime', "1 Year Time Span of Air Concentrations")
plt.matshow(df.corr())
plt.colorbar()
plt.show()
print(df.corr())
from sklearn.model_selection import train_test_split
X=df[['CO_GT', 'PT08_S1_CO', 'C6H6_GT', 'PT08_S2_NMHC', 'Nox_GT', 'PT08_S3_Nox', 'NO2_GT', 'PT08_S4_NO2', 'PT08_S5_O3', 'T']]
y=df[['RH','AH']]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3)
from sklearn.linear_model import LinearRegression
reg=LinearRegression(normalize="Boolean")
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)
print(y_pred)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))
print(reg.intercept_)
print(reg.coef_)
X=df[['CO_GT', 'PT08_S1_CO', 'C6H6_GT', 'PT08_S2_NMHC', 'Nox_GT', 'PT08_S3_Nox', 'NO2_GT', 'PT08_S4_NO2', 'PT08_S5_O3', 'T']]
y_pred_all=reg.predict(X)
df['RH_pred']=y_pred_all[:,0]
df['AH_pred']=y_pred_all[:,1]
plt.figure(figsize=(22,15))
plt.plot_date(df.datetime, df.RH, marker='.', label="True", alpha=0.5)
plt.plot_date(df.datetime, df.RH_pred, marker='.', label="Predicted")
plt.title("Comparison of True and Predicted values of Relative Humidity at various times", fontsize=20)
plt.xlabel("Datetime", fontsize=20)
plt.ylabel("Relative Humidity", fontsize=20)
plt.legend(fontsize=15, facecolor='white', markerscale=2)
plt.figure(figsize=(22,15))
plt.plot_date(df.datetime, df.AH, marker='.', label="True", alpha=0.5)
plt.plot_date(df.datetime, df.AH_pred, marker='.', label="Predicted")
plt.title("Comparison of True and Predicted values of Relative Humidity at various times", fontsize=20)
plt.xlabel("Datetime", fontsize=20)
plt.ylabel("Relative Humidity", fontsize=20)
plt.legend(fontsize=15, facecolor='white', markerscale=2)
n_estimators=[300]
max_depth=[2,3,4,6,7]
booster=['gbtree']
learning_rate=[0.03, 0.06, 0.1, 0.15, 0.2]
min_child_weight=[4, 5]
base_score=[0.2,0.25, 0.5, 0.75]
hyperparameter_grid={'n_estimators':n_estimators,
                     'max_depth':max_depth,
                     'learning_rate':learning_rate,
                     'min_child_weight':min_child_weight,
                     'booster':booster,
                     'base_score':base_score}
import xgboost
xreg=xgboost.XGBRegressor()
xreg1=xgboost.XGBRegressor()
y1=df[['RH']]
X_train, X_test, y1_train, y1_test=train_test_split(X, y1, test_size=0.2)
from sklearn.model_selection import RandomizedSearchCV
random_cv1=RandomizedSearchCV(estimator=xreg,
                             param_distributions=hyperparameter_grid,
                             n_iter=10,cv=5,scoring='neg_mean_squared_error')
random_cv1.fit(X_train, y1_train)
print(random_cv1.best_estimator_)
xreg=xgboost.XGBRegressor(base_score=0.2, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.1, max_delta_step=0, max_depth=5,
             min_child_weight=2, monotone_constraints=None,
             n_estimators=1200, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
             validate_parameters=False, verbosity=None)
xreg.fit(X_train, y1_train)
y1_pred=xreg.predict(X_test)
print(y1_pred)
print(r2_score(y1_test, y1_pred))
df['RH_pred_xg']=xreg.predict(X)
plt.figure(figsize=(22,15))
plt.plot_date(df.datetime, df.RH, marker='.', label="True")
plt.plot_date(df.datetime, df.RH_pred_xg, marker='.', label="Predicted")
plt.title("Comparison of True and Predicted values of Relative Humidity at various times (with boosting)", fontsize=20)
plt.xlabel("Datetime", fontsize=20)
plt.ylabel("Relative Humidity", fontsize=20)
plt.legend(fontsize=15, facecolor='white', markerscale=2)
y2=df[['AH']]
X_train, X_test, y2_train, y2_test=train_test_split(X, y2, test_size=0.3)
xreg1=xgboost.XGBRegressor()
random_cv2=RandomizedSearchCV(estimator=xreg,
                             param_distributions=hyperparameter_grid,
                             n_iter=10,cv=5,scoring='neg_mean_squared_error')
random_cv2.fit(X_train, y2_train)
print(random_cv2.best_estimator_)
xreg1=xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.06, max_delta_step=0, max_depth=5,
             min_child_weight=4, monotone_constraints=None,
             n_estimators=1000, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
             validate_parameters=False, verbosity=None)
xreg1.fit(X_train, y2_train)
y2_pred=xreg1.predict(X_test)
print(y2_pred)
print(r2_score(y2_test, y2_pred))
df['AH_pred_xg']=xreg1.predict(X)
plt.figure(figsize=(22,15))
plt.plot_date(df.datetime, df.AH, marker='.', label="True")
plt.plot_date(df.datetime, df.AH_pred_xg, marker='.', label="Predicted")
plt.title("Comparison of True and Predicted values of Absolute Humidity at various times (with boosting)", fontsize=20)
plt.xlabel("Datetime", fontsize=20)
plt.ylabel("Absolute Humidity", fontsize=20)
plt.legend(fontsize=15, facecolor='white', markerscale=2)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
print(linreg.coef_)
print(linreg.intercept_)
print(y_train.shape)
from sklearn import metrics
import numpy as np
def typical_linear_model_performance(y_pred):
    print ('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print ('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print ('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
from sklearn.model_selection import cross_val_score
def get_cross_value_score(model):
    scores = cross_val_score(model, X_train, y_train,cv=5,scoring='r2')
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))
    print('\n')
get_cross_value_score(linreg)
typical_linear_model_performance(y_pred)
from sklearn.neighbors import KNeighborsRegressor
clf_knn = KNeighborsRegressor(n_neighbors=10)
clf_knn = clf_knn.fit(X_train,y_train)
y_pred = clf_knn.predict(X_test)
get_cross_value_score(clf_knn)
typical_linear_model_performance(y_pred)
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
dt_one_reg=DecisionTreeRegressor()
dt_model=dt_one_reg.fit(X_train,y_train)
y_pred_dtone=dt_model.predict(X_test)
print('RMSE of Decision Tree Regression:',np.sqrt(mean_squared_error(y_pred_dtone,y_test)))
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor()
rf_model=rf_reg.fit(X_train,y_train)
y_pred_rf=rf_model.predict(X_test)
print('RMSE of predicted RH in RF model:',np.sqrt(mean_squared_error(y_test,y_pred_rf)))
rf_params={'n_estimators':[10,20],'max_depth':[8,10],'max_leaf_nodes':[70,90]}
rf_grid=GridSearchCV(rf_reg,rf_params,cv=10)
rf_model_two=rf_grid.fit(X_train,y_train)
y_pred_rf_two=rf_model_two.predict(X_test)
print('RMSE using RF grid search method',np.sqrt(mean_squared_error(y_test,y_pred_rf_two)))
print(df.info())
features=list(df.columns)
features.remove('datetime')
features.remove('PT08_S4_NO2')
features.remove('RH_pred')
features.remove('AH_pred')
features.remove('AH_pred_xg')
features.remove('RH_pred_xg')
features.remove('week_day')
features.remove('CO_level')
X = df[features]
y = df['C6H6_GT']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=0)
print(X_train.shape, y_train.shape)
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1000, 
                               criterion='mse', 
                               random_state=1, 
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predictions = regressor.predict(X_test)
print('R-squared:', regressor.score(X_test, y_test))
scores = cross_val_score(regressor, X, y, cv=5)
print ("Average of scores: ", scores.mean())
print ("Cross validation scores: ", scores)
from sklearn.model_selection import *
from sklearn.model_selection import KFold
def train_and_evaluate(clf, X_train, y_train):
    
    clf.fit(X_train, y_train)
    
    print ("Coefficient of determination on training set:",clf.score(X_train, y_train))
    
    cv = KFold(5, shuffle=True, random_state=33)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print ("Average coefficient of determination using 5-fold crossvalidation:",np.mean(scores))
from sklearn import linear_model
clf_sgd = linear_model.SGDRegressor(loss='squared_loss', penalty=None,  random_state=42)
train_and_evaluate(clf_sgd,X_train,y_train)
print(clf_sgd.coef_)
clf_sgd1 = linear_model.SGDRegressor(loss='squared_loss', penalty='l2',  random_state=42)
train_and_evaluate(clf_sgd1,X_train,y_train)
clf_sgd2 = linear_model.SGDRegressor(loss='squared_loss', penalty='l1',  random_state=42)
train_and_evaluate(clf_sgd2,X_train,y_train)
from sklearn import ensemble
clf_et=ensemble.ExtraTreesRegressor(n_estimators=10,random_state=42)
train_and_evaluate(clf_et,X_train,y_train)
imp_features = (np.sort((clf_et.feature_importances_,features),axis=0))
for rank,f in zip(imp_features[0],imp_features[1]):
    print("{0:.3f} <-> {1}".format(float(rank), f))
from sklearn import metrics
def measure_performance(X,y,clf, show_accuracy=True,
                        show_classification_report=True,
                        show_confusion_matrix=True,
                        show_r2_score=False):
    y_pred=clf.predict(X)   
    if show_accuracy:
        print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)),"\n")
    if show_classification_report:
        print ("Classification report")
        print (metrics.classification_report(y,y_pred),"\n")
    if show_confusion_matrix:
        print ("Confusion matrix")
        print (metrics.confusion_matrix(y,y_pred),"\n")
    if show_r2_score:
        print ("Coefficient of determination:{0:.3f}".format(metrics.r2_score(y,y_pred)),"\n")
measure_performance(X_test,y_test,clf_et,
                    show_accuracy=False,
                    show_classification_report=False,
                    show_confusion_matrix=False,
                    show_r2_score=True)