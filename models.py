from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import streamlit as st

def ols_model(X, y):
	model = sm.OLS(y, X).fit()
	predictions = model.predict(X)
	st.write("OLS")
	summary = model.summary()
	st.write(summary)
	return model

def dt_regression(X, y, max_depth):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
	model = DecisionTreeRegressor(max_depth=max_depth)
	model.fit(X_train, y_train)
	pred_results = model.predict(X_test)
	result = regression(y_test, pred_results, model)
	st.write("Evaluation Metrics ", result)
	dot_data = export_graphviz(model, filled=True, rounded=True, out_file=None)
	st.graphviz_chart(dot_data)
	return

def regression(y_test, pred_results, model):
   result = {
      'R2': r2_score(y_test, pred_results),
      'MSE': mean_squared_error(y_test, pred_results),
      'MAE': mean_absolute_error(y_test, pred_results)
   }
   return(result)