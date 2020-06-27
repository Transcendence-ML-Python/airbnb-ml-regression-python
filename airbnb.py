import pandas as pd
import streamlit as st
import seaborn as sns
import models
from sklearn.model_selection import train_test_split

def airbnb_regression_prep():
    airbnb_data = pd.read_csv('AB_NYC_2019.csv')
    raw_data = airbnb_data
    # sns.catplot(x="room_type", y="price", data=raw_data)
    # st.pyplot()
    # sns.catplot(x="neighbourhood", y="price", data=raw_data)
    # st.pyplot()
    # sns.catplot(x="neighbourhood_group", y="price", data=raw_data)
    # st.pyplot()
    # sns.catplot(x="neighbourhood", y="number_of_reviews", data=raw_data)
    # st.pyplot()
    # sns.catplot(x="neighbourhood_group", y="number_of_reviews", data=raw_data)
    # st.pyplot()
    data = airbnb_data.sort_values(["neighbourhood", "room_type"], ascending=(True, True))
    data1 = data.groupby(["neighbourhood", "room_type"])["price"].mean()
    merged_left = pd.merge(left=data, right=data1, how='left', left_on=("neighbourhood", "room_type"),
                           right_on=("neighbourhood", "room_type"))
    merged_left['price_y'] = merged_left['price_y'].astype('category')
    X = merged_left[['room_type', 'neighbourhood_group', 'neighbourhood']]
    y = merged_left[['price_y']]

    X = pd.get_dummies(X)
    return (X, y)

X, Y = airbnb_regression_prep()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
model = ["-----","OLS", "Decision Tree Regression"]
option = st.sidebar.selectbox('Machine Learning Model', model)
if option == "OLS":
    model = models.ols_model(X, Y)
if option == "Decision Tree Regression":
    model = models.dt_regression(X, Y,max_depth=20)
else:
    pass
