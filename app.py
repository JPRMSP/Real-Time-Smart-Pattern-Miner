import streamlit as st
import itertools
import math
import multiprocessing as mp
import pandas as pd
import random

st.set_page_config(page_title="Real-Time Smart Pattern Miner", layout="wide")

st.title("🧠 Real-Time Smart Pattern Miner")
st.write("🚀 A live data mining web app implementing Apriori, Association Rules, Classification, and Clustering — without datasets or pretrained models.")

# ---------------------------------------
# 🔥 Frequent Pattern Mining (Apriori)
# ---------------------------------------
def apriori(transactions, min_support=0.5):
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1
    total_transactions = len(transactions)
    freq_items = {frozenset([item]): count/total_transactions for item, count in item_counts.items() if count/total_transactions >= min_support}

    k = 2
    results = dict(freq_items)
    while True:
        candidates = [frozenset(set1 | set2) for set1 in results.keys() for set2 in results.keys() if len(set1 | set2) == k]
        candidates = set(candidates)
        new_freq = {}
        for cand in candidates:
            count = sum(1 for t in transactions if cand.issubset(t))
            support = count / total_transactions
            if support >= min_support:
                new_freq[cand] = support
        if not new_freq:
            break
        results.update(new_freq)
        k += 1
    return results

# ---------------------------------------
# 📊 Association Rule Generation
# ---------------------------------------
def generate_rules(freq_itemsets, min_confidence=0.6):
    rules = []
    for itemset, support in freq_itemsets.items():
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for antecedent in itertools.combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    conf = support / freq_itemsets.get(antecedent, 1e-9)
                    if conf >= min_confidence:
                        rules.append((set(antecedent), set(consequent), round(support, 2), round(conf, 2)))
    return rules

# ---------------------------------------
# 🧬 K-Means Clustering
# ---------------------------------------
def kmeans(data, k=2, max_iter=100):
    centroids = random.sample(data, k)
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [math.dist(point, c) for c in centroids]
            clusters[distances.index(min(distances))].append(point)
        new_centroids = [tuple(sum(coords) / len(coords) for coords in zip(*cluster)) if cluster else centroids[i] for i, cluster in enumerate(clusters)]
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return clusters, centroids

# ---------------------------------------
# ⚙️ Parallel Processing
# ---------------------------------------
def parallel_apriori(transactions):
    return apriori(transactions, min_support=0.5)

def parallel_kmeans(data):
    return kmeans(data, k=2)

# ---------------------------------------
# 🧪 UI – Frequent Pattern Mining
# ---------------------------------------
st.header("📦 Frequent Pattern Mining & Association Rules")
raw_input = st.text_area("Enter transactions (comma-separated items per line):", "milk,bread\nmilk,diaper,bread\nmilk,bread,cola")
transactions = [frozenset(t.strip().split(",")) for t in raw_input.strip().split("\n") if t]

if st.button("🚀 Run Apriori & Rules in Parallel"):
    with mp.Pool(2) as pool:
        freq_future = pool.apply_async(parallel_apriori, (transactions,))
        freq_itemsets = freq_future.get()

    st.subheader("✅ Frequent Itemsets")
    st.write(pd.DataFrame([(list(k), round(v, 2)) for k, v in freq_itemsets.items()], columns=["Itemset", "Support"]))

    st.subheader("🔗 Association Rules")
    rules = generate_rules(freq_itemsets)
    st.write(pd.DataFrame(rules, columns=["Antecedent", "Consequent", "Support", "Confidence"]))

# ---------------------------------------
# 🧪 UI – K-Means Clustering
# ---------------------------------------
st.header("📊 K-Means Clustering")
raw_points = st.text_area("Enter 2D points (comma-separated x,y per line):", "1,2\n2,3\n8,8\n9,9")
points = [tuple(map(float, p.strip().split(","))) for p in raw_points.strip().split("\n") if p]

if st.button("⚙️ Run Parallel K-Means"):
    with mp.Pool(2) as pool:
        kmeans_future = pool.apply_async(parallel_kmeans, (points,))
        clusters, centroids = kmeans_future.get()

    st.subheader("📍 Cluster Centroids")
    st.write(centroids)

    for i, cluster in enumerate(clusters):
        st.write(f"Cluster {i+1}: {cluster}")

# ---------------------------------------
# 🧪 Simple Rule-Based Classification
# ---------------------------------------
st.header("🧠 Simple Rule-Based Text Classifier")
text_input = st.text_area("Enter text to classify:", "This transaction includes milk and bread.")
if st.button("📌 Classify Text"):
    if "milk" in text_input and "bread" in text_input:
        st.success("🍞 Classified as: Grocery Shopping")
    elif "cola" in text_input:
        st.info("🥤 Classified as: Beverage Purchase")
    else:
        st.warning("🤔 Classified as: Unknown Category")
