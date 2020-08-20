#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor

img = Image.open('C:/Users/admin/Downloads/ji.jpg')
st.image(img)
st.markdown("# **Welcome to Vaishnavi's Streamlit Application**")

df = pd.read_csv("Materials.csv")

def get_formation_energy_plot(df_formation, spin_config):
    """Plot the heatmap for formation energy for different configurations"""
    
    TMlist = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn', 'Y' ,'Zr','Nb','Mo','Tc','Ru','Rh','Pd', 'Ag','Cd', 'W' ,'Ir','Pt','Au']
    Xlist = ['F','Cl','Br','I']
    data = df_formation[df_formation.spin_configuration == spin_config].pivot(index='Atomic_number_X', columns='Atomic_number_A', values='formation_energy')
    if spin_config == 'spin':
        title = 'Ferromagnetic'
    if spin_config == 'spin_so':
        title = 'Ferromagnetic Spin Orbit Coupling'
    if spin_config == 'afm':
        title = 'Anti-Ferromagnetic'
    if spin_config == 'afm_so':
        title = 'Anti-Ferromagnetic Spin Orbit Coupling'
    if spin_config == 'initial':
        title = 'Initial'
    
    plt.axes().set_title(title)
    mask = np.zeros_like(data)
    mask[np.triu_indices_from(mask)] = True
    sb.set(font_scale = 0.5)
    heat_map = sb.heatmap(data, vmax=0, xticklabels=TMlist, yticklabels=Xlist,square = True, cmap='inferno', cbar_kws = {'shrink':0.25}, mask = data.isnull()).set_facecolor('xkcd:gray')
    st.pyplot()



st.sidebar.markdown("## 2D Magnetic Materials")
st.sidebar.markdown("Select the checkboxes to get more options")

if  st.sidebar.checkbox("Formation Energy"):
    st.markdown("## Formation Energy")
    st.sidebar.markdown("Select one or multiple checkboxes to view the heatmaps for the formation energy for 2D magentic materials. The heatmaps shown are according to the spin configurations of the crystal structure")
    if st.sidebar.checkbox("Formation energy for Initial Configuration of Materials"):
        get_formation_energy_plot(df, 'initial')
    if st.sidebar.checkbox("Formation energy for Anti-Ferromagnetic Materials"):
        get_formation_energy_plot(df, 'afm')
    if st.sidebar.checkbox("Formation energy for Ferromagnetic Materials"):
        get_formation_energy_plot(df, 'spin')
    if st.sidebar.checkbox("Formation energy for Anti-Ferromagnetic Spin-Orbit Coupled Materials"):
        get_formation_energy_plot(df, 'afm_so')
    if st.sidebar.checkbox("Formation energy for Ferromagnetic Spin-Orbit Coupled Materials"):
        get_formation_energy_plot(df, 'spin_so')




test = df[['formation_energy','electronegativity', "dipole_polarizability", "valence_electrons"]]
test = test.dropna()
test = test[test.formation_energy<100]

test_size = st.sidebar.slider('Select Train Test Split Percentage', 1, 100, 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(test.loc[:, test.columns != 'formation_energy'], test.formation_energy,
                                                    test_size=test_size, random_state=42)

regr = st.sidebar.selectbox("Please choose a Regression Algorithm", ('None', 'Linear Regression', 'Lasso Regression', 'Kernel Ridge Regression', 'Random Forest Regression'))

def regression_algo_selection(df, regr):
    if regr == 'Linear Regression':
        lm = linear_model.LinearRegression()
        lm.fit(X_train, y_train)
        predictions = lm.predict(X_test)
        plot_regr(lm, predictions, y_test)

    if regr == 'Lasso Regression':
        alpha = st.sidebar.slider('Select alpha', 0.00001, 1.0, 0.001)
        lm = linear_model.Lasso(alpha = alpha)
        lm.fit(X_train, y_train)
        predictions = lm.predict(X_test)
        plot_regr(lm, predictions, y_test)

    if regr == 'Kernel Ridge Regression':
        kernel = st.sidebar.radio('Select kernel', ('linear', 'polynomial', 'rbf'))
        alpha = st.sidebar.slider('Select alpha', 0, 10, 1)
        lm = KernelRidge(kernel = kernel, alpha = alpha)
        lm.fit(X_train, y_train)
        predictions = lm.predict(X_test)
        plot_regr(lm, predictions, y_test)

    if regr == 'Random Forest Regression':
        max_depth = st.sidebar.slider('Select Maximum Depth', 2, 10, 2)
        n_estimators = st.sidebar.slider('Select Number of Estimators', 10, 100, 5)
        min_samples_split = st.sidebar.slider('Select Minimum Sample Split', 2, 50, 2)
        lm = RandomForestRegressor(max_depth= max_depth, n_estimators =n_estimators, min_samples_split =min_samples_split)
        lm.fit(X_train, y_train)
        predictions = lm.predict(X_test)
        plot_regr(lm, predictions, y_test)

def plot_regr(model, predictions, y_test):
    plt.figure(figsize = (4,2))
    plt.scatter(predictions, y_test, s = 5)
    st.pyplot()
    st.text(f"The R-squared Value for the Model is {model.score(X_test, y_test)}")


regression_algo_selection(df, regr)



