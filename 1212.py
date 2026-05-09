import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---------------- DATA ----------------
X = np.array([
    [1000], [5000], [12000],
    [1200], [5200], [12500],
    [1500], [5500], [13000],
    [1800], [6000], [14000]
])

# Model
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
kmeans.fit(X)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

# ---------------- UI ----------------
st.set_page_config(page_title="Customer Segmentation", page_icon="📊")

st.title("📊 Customer Segmentation App (K-Means)")
st.write("Monthly Spending অনুযায়ী customer group করা হচ্ছে")

# User input
spending = st.slider("Enter Monthly Spending", 0, 20000, 5000)

# Predict cluster
cluster = kmeans.predict([[spending]])

# Result
st.subheader("Result")

st.success(f"Customer belongs to Group: {cluster[0]}")

# ---------------- GRAPH ----------------
st.subheader("Visualization")

fig, ax = plt.subplots()

ax.scatter(X, np.zeros_like(X), c=labels, s=120)

ax.scatter(centers, np.zeros_like(centers),
           color='red', s=200, marker='x', label='Centers')

ax.scatter(spending, 0,
           color='black', s=200, label='You')

ax.set_xlabel("Monthly Spending")
ax.set_yticks([])
ax.set_title("Customer Segmentation using K-Means")
ax.legend()

st.pyplot(fig)