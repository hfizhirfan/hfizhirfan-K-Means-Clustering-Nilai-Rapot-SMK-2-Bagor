import pandas as pd
from sklearn.cluster import KMeans 
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template
app=Flask(__name__)
def myfunc(a, b):
    return {"x":a,"y":b}
@app.route('/')
def home():  
    return render_template('index.html')
@app.route('/data')
def data():  
    return render_template('data.html')
@app.route('/kelasX')
def kelasX(): 
    try:
        dataset = pd.read_csv("KelasX.csv") 
        return render_template('kelasX.html', tables=[dataset.to_html(classes='table', border=0, header="true")], titles=[''])
    except Exception as e:
        print(f"Error: {e}")
        return "Terjadi kesalahan. Periksa konsol untuk detailnya."
@app.route('/kelasXI')
def kelasXI(): 
    try:
        dataset = pd.read_csv("KelasXI.csv") 
        return render_template('kelasXI.html', tables=[dataset.to_html(classes='table', border=0, header="true")], titles=[''])
    except Exception as e:
        print(f"Error: {e}")
        return "Terjadi kesalahan. Periksa konsol untuk detailnya."
@app.route('/kelasXII')
def kelasXII(): 
    try:
        dataset = pd.read_csv("KelasXII.csv") 
        return render_template('kelasXII.html', tables=[dataset.to_html(classes='table', border=0, header="true")], titles=[''])
    except Exception as e:
        print(f"Error: {e}")
        return "Terjadi kesalahan. Periksa konsol untuk detailnya."
@app.route('/elbow')
def elbow():   
    markers={}
    dataset = pd.read_csv("bdns2.csv")
    x = dataset.iloc[:, [1,2,3,4,5,6]].values
    wcss = []
    for i in range (2, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    markers['elbow'] = wcss 
    return render_template('elbow.html',markers=markers )

@app.route('/chart')
def chart():   
    markers={}
    dataset = pd.read_csv("bdns2.csv")
    x = dataset.iloc[:, [1,2,3,4,5,6]].values
    kota = dataset.iloc[:,[0]].values
    kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(x)
    markers['tinggi'] = list(map(myfunc,x[y_kmeans == 2, 0],x[y_kmeans == 2, 1]))
    markers['sedang'] = list(map(myfunc,x[y_kmeans == 1, 0],x[y_kmeans == 1, 1]))
    markers['rendah'] = list(map(myfunc,x[y_kmeans == 0, 0],x[y_kmeans == 0, 1]))
    markers['centroid'] = list(map(myfunc,kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1]))
    return render_template('chart.html',markers=markers )
if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)