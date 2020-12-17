#coding:gbk
"""
���þ������㷨���з���
���ߣ����˷�
���ڣ�2020/12/16
"""
import pandas as pd           # ������Ҫ�õĿ�
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sb

# ��������
df = pd.read_csv('frenchwine.csv')
df.columns = ['species', 'alcohol', 'malic_acid', 'ash', 'alcalinity ash', 'magnesium']
# �鿴ǰ5������
df.head()
print(df.head()) 


# �鿴����������ͳ����Ϣ
df.describe()
print(df.describe())

def scatter_plot_by_category(feat, x, y): #���ݵĿ��ӻ� 
    alpha = 0.5
    gs = df.groupby(feat)
    cs = cm.rainbow(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1][x], g[1][y], color=c, alpha=alpha)

plt.figure(figsize=(20,5))
plt.subplot(131)
scatter_plot_by_category('species', 'alcohol', 'ash')
plt.xlabel('alcohol')
plt.ylabel('ash')
plt.title('species')
plt.show()

plt.figure(figsize=(20, 10)) #����seaborn���������Iris����ͬ����ͼ
for column_index, column in enumerate(df.columns):
    if column == 'species':
        continue
    plt.subplot(2, 2, column_index + 1)
    sb.violinplot(x='species', y=column, data=df)
plt.show()

# ���ȶ����ݽ����з֣������ֳ�ѵ�����Ͳ��Լ�
from sklearn.model_selection import train_test_split #����sklearn���н�����飬����ѵ�����Ͳ��Լ�
all_inputs = df[['alcohol', 'malic_acid', 'ash', 'alcalinity ash', 'magnesium']].values
all_species = df['species'].values

(X_train,
 X_test,
 Y_train,
 Y_test) = train_test_split(all_inputs, all_species, train_size=0.7, random_state=1)#80%������ѡΪѵ����
 
 


# ʹ�þ������㷨����ѵ��
from sklearn.tree import DecisionTreeClassifier #����sklearn���е�DecisionTreeClassifier������������
# ����һ������������
decision_tree_classifier = DecisionTreeClassifier()

# ѵ��ģ��
model = decision_tree_classifier.fit(X_train, Y_train)
# ���ģ�͵�׼ȷ��
print(decision_tree_classifier.score(X_test, Y_test)) 


# ʹ��ѵ����ģ�ͽ���Ԥ�⣬Ϊ�˷��㣬
# ����ֱ�ӰѲ��Լ�����������ó���������ʵ��ʹ��ʱ�����������в��Լ����ݼ�X_test��ģ�ͽ��в��ԡ�
X_test = [(13.42,3.21,2.62,23.5,95), ( 12.32,2.77,2.37,22,90), ( 13.75,1.59,2.7,19.5,135)]
print(X_test)#����3�����ݽ��в��ԣ���ȡ3��������Ϊģ�͵������
m = model.predict(X_test)
for a in m:
    if a == "Zinfandel":
        print("�ɷ���")
    elif a == "Syrah":
        print("����")
    elif a == "Sauvignon":
        print("��ϼ��")
#������ԵĽ���������ģ��Ԥ��Ľ��


##���������ӻ�
from IPython.display import Image
# from sklearn.externals.six import StringIO  #sklearn 0.23�汾�Ѿ�ɾ���������,ֱ�Ӱ�װsix����
from six import StringIO
from sklearn.tree import export_graphviz


features = list(df.columns[:-1])
print(features)


import pydotplus
import os #Ҫ��װһ��Graphviz����
os.environ['PATH'] = os.environ['PATH'] + (';c:\\Program Files\\Graphviz\\bin\\') #
dot_data = StringIO()
export_graphviz(decision_tree_classifier, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph[0].create_png())
graph.write_pdf("frenchwine.pdf") #��iris���ݼ����þ������㷨���ӻ�������ֵ�iris.pdf�ļ���


