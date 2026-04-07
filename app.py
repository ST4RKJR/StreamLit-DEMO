import streamlit as st
from data import generate_data
from tree import build_tree
from predict import predict_one
from visualize import plot_data, plot_split, plot_decision_boundary
from split import best_split
from utils import format_path
import numpy as np

st.title("🌳 Decision Tree Visualizer")

# Sidebar
st.sidebar.header("Controls")
noise = st.sidebar.slider("Noise", 0.0, 0.5, 0.1)
max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
min_samples = st.sidebar.slider("Min Samples", 1, 20, 5)

# Session state for data
if "X" not in st.session_state:
    st.session_state.X, st.session_state.y = generate_data(n=200, noise=noise)

if st.sidebar.button("New Data"):
    st.session_state.X, st.session_state.y = generate_data(n=200, noise=noise)

X, y = st.session_state.X, st.session_state.y

# Build tree
tree = build_tree(X, y, max_depth=max_depth, min_samples=min_samples)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Data",
    "Best Split",
    "Decision Boundary",
    "Prediction Path"
])

# Tab 1
with tab1:
    st.subheader("Dataset")
    fig = plot_data(X, y)
    st.pyplot(fig)

# Tab 2
with tab2:
    st.subheader("Best Split")
    f, t, g = best_split(X, y)
    st.write(f"Feature: {f}, Threshold: {t:.2f}, Gain: {g:.4f}")
    fig = plot_split(X, y, f, t)
    st.pyplot(fig)

# Tab 3
with tab3:
    st.subheader("Decision Boundary")
    fig = plot_decision_boundary(X, y, tree, predict_one)
    st.pyplot(fig)

# Tab 4
with tab4:
    st.subheader("Prediction Path")

    x1 = st.slider("Feature 1", float(X[:,0].min()), float(X[:,0].max()))
    x2 = st.slider("Feature 2", float(X[:,1].min()), float(X[:,1].max()))

    pred, path = predict_one(tree, np.array([x1, x2]))

    st.write("Prediction:", pred)
    st.write("Path:", format_path(path))

# Debug (optional)
st.write("X shape:", X.shape)
st.write("y shape:", y.shape)
st.write("Noise:", noise)
st.write("Max Depth:", max_depth)
st.write("Min Samples:", min_samples)