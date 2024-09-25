import tkinter as tk
from tkinter import ttk
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

def show_frame(data, n=15):
    root = tk.Tk()
    root.title("Tabela")

    frame = ttk.Frame(root)
    frame.pack(fill='both', expand=True)

    canvas = tk.Canvas(frame)
    canvas.pack(side='left', fill='both', expand=True)

    scroll_y = tk.Scrollbar(frame, orient='vertical', command=canvas.yview)
    scroll_y.pack(side='right', fill='y')

    scroll_x = tk.Scrollbar(root, orient='horizontal', command=canvas.xview)
    scroll_x.pack(side='bottom', fill='x')

    tree_frame = ttk.Frame(canvas)
    tree_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    canvas.create_window((0, 0), window=tree_frame, anchor='nw')
    canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

    style = ttk.Style()
    style.configure("Treeview", anchor='center')

    tree = ttk.Treeview(tree_frame, columns=list(data.columns), show='headings', height=n)

    for col in data.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor='center')

    for index, row in data.iterrows():
        tree.insert("", "end", values=list(row))

    tree.pack(fill='both', expand=True)

    scrollbar_y_tree = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar_y_tree.set)
    scrollbar_y_tree.pack(side='right', fill='y')

    root.mainloop()
def visualize_barplot(column_name,data,value='price'):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=column_name, y=value, data=data, estimator='mean')
    plt.xlabel(column_name)
    plt.ylabel('mean_'+value)
    plt.xticks(rotation=45)
    plt.show()

def visualize_scatter(column_name,data,value = 'price'):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=column_name, y=value, data=data)
    plt.xlabel(column_name)
    plt.ylabel(value)
    plt.xticks(rotation=45)
    plt.show()

def top_5_words(slowa,n=5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=n)
    models_v = vectorizer.fit_transform(slowa)
    words = vectorizer.get_feature_names_out()
    tfidf_sums = np.sum(models_v.toarray(), axis=0)
    sorted_words = sorted(zip(words, tfidf_sums), key=lambda x: x[1], reverse=True)
    top_10_words = sorted_words[:n]
    top_10_words = [i[0] for i in top_10_words]
    return top_10_words

def count_words(data,column):
    from collections import Counter
    all_words = data[column].explode().str.split().sum()
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common()
    return most_common_words