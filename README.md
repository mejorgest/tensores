# tensores

import os
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
from matplotlib import colors as mcolors
import warnings

from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

from pandas import DataFrame
import seaborn as sns

class Analisis_Predictivo:

    def __init__(self,datos:DataFrame, predecir:str, predictoras = [],
                 modelo = None,train_size = 80,random_state = None):
        '''
        datos: Datos completos y listos para construir un modelo
        
        modelo: Instancia de una Clase de un método de clasificación(KNN,Árboles,SVM,etc).
        Si no especifica un modelo no podrá utilizar el método fit_n_review()
        
        predecir: Nombre de la variable a predecir
        
        predictoras: Lista de los nombres de las variables predictoras.
        Si vacío entonces utiliza todas las variables presentes excepto la variable a predecir.
        
        train_size: Proporción de la tabla de entrenamiento respecto a la original.
        
        random_state: Semilla aleatoria para la división de datos(training-testing).
        '''        
        self.__datos = datos
        self.__predecir = predecir
        self.__predictoras = predictoras
        self.__modelo = modelo
        self.__random_state = random_state
        if modelo != None:
            self.__train_size = train_size
            self._training_testing()
    
    @property
    def datos(self):
        return self.__datos
    
    @property
    def predecir(self):
        return self.__predecir
    
    @property
    def predictoras(self):
        return self.__predictoras
    
    @property
    def modelo(self):
        return self.__modelo
    
    @property
    def random_state(self):
        return self.__random_state
    
    @property
    def train_size(self):
        return self.__train_size
    
    @datos.setter
    def datos(self, datos):
        self.__datos = datos
    
    @predecir.setter
    def predecir(self, predecir):
        self.__predecir = predecir
        
    @predictoras.setter
    def predictoras(self, predictoras):
        self.__predictoras = predictoras
        
    @modelo.setter
    def modelo(self, modelo):
        self.__modelo = modelo
        
    @random_state.setter
    def random_state(self, random_state):
        self.__random_state = random_state
        
    @train_size.setter
    def train_size(self, train_size):
        self.__train_size = train_size
        
    def _training_testing(self):
        if len(self.predictoras) == 0:
            X = self.datos.drop(columns=[self.predecir])
        else:
            X = self.datos[self.predictoras]
            
        y = self.datos[self.predecir].values
        
        train_test = train_test_split(X, y, train_size=self.train_size, 
                                      random_state=self.random_state)
        self.X_train, self.X_test,self.y_train, self.y_test = train_test
        
    def fit_predict(self):
        if(self.modelo != None):
            self.modelo.fit(self.X_train,self.y_train)
            return self.modelo.predict(self.X_test)
        
    def fit_predict_resultados(self, imprimir = True):
        if(self.modelo != None):
            y = self.datos[self.predecir].values
            prediccion = self.fit_predict()
            MC = confusion_matrix(self.y_test, prediccion)
            indices = self.indices_general(MC,list(np.unique(y)))
            if imprimir == True:
                for k in indices:
                    print("\n%s:\n%s"%(k,str(indices[k])))
            
            return indices
    
    def indices_general(self,MC, nombres = None):
        "Método para calcular los índices de calidad de la predicción"
        precision_global = np.sum(MC.diagonal()) / np.sum(MC)
        error_global = 1 - precision_global
        precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
        if nombres!=None:
            precision_categoria.columns = nombres
        return {"Matriz de Confusión":MC, 
                "Precisión Global":precision_global, 
                "Error Global":error_global, 
                "Precisión por categoría":precision_categoria}
    
    def distribucion_variable_predecir(self):
        "Método para graficar la distribución de la variable a predecir"
        variable_predict = self.predecir
        data = self.datos
        colors = list(dict(**mcolors.CSS4_COLORS))
        df = pd.crosstab(index=data[variable_predict],columns="valor") / data[variable_predict].count()
        fig = plt.figure(figsize=(10,9))
        g = fig.add_subplot(111)
        countv = 0
        titulo = "Distribución de la variable %s" % variable_predict
        for i in range(df.shape[0]):
            g.barh(1,df.iloc[i],left = countv, align='center',color=colors[11+i],label= df.iloc[i].name)
            countv = countv + df.iloc[i]
        vals = g.get_xticks()
        g.set_xlim(0,1)
        g.set_yticklabels("")
        g.set_title(titulo)
        g.set_ylabel(variable_predict)
        g.set_xticklabels(['{:.0%}'.format(x) for x in vals])
        countv = 0 
        for v in df.iloc[:,0]:
            g.text(np.mean([countv,countv+v]) - 0.03, 1 , '{:.1%}'.format(v), color='black', fontweight='bold')
            countv = countv + v
        g.legend(loc='upper center', bbox_to_anchor=(1.08, 1), shadow=True, ncol=1)
        
    def poder_predictivo_categorica(self, var:str):
        "Método para ver la distribución de una variable categórica respecto a la predecir"
        data = self.datos
        variable_predict = self.predecir
        df = pd.crosstab(index= data[var],columns=data[variable_predict])
        df = df.div(df.sum(axis=1),axis=0)
        titulo = "Distribución de la variable %s según la variable %s" % (var,variable_predict)
        g = df.plot(kind='barh',stacked=True,legend = True, figsize = (10,9), \
                    xlim = (0,1),title = titulo, width = 0.8)
        vals = g.get_xticks()
        g.set_xticklabels(['{:.0%}'.format(x) for x in vals])
        g.legend(loc='upper center', bbox_to_anchor=(1.08, 1), shadow=True, ncol=1)
        for bars in g.containers:
            plt.setp(bars, width=.9)
        for i in range(df.shape[0]):
            countv = 0 
            for v in df.iloc[i]:
                g.text(np.mean([countv,countv+v]) - 0.03, i , '{:.1%}'.format(v), color='black', fontweight='bold')
                countv = countv + v
                
                
    def poder_predictivo_numerica(self,var:str):
        "Función para ver la distribución de una variable numérica respecto a la predecir"
        sns.FacetGrid(self.datos, hue=self.predecir, height=6).map(sns.kdeplot, var, shade=True).add_legend()

#ejemplosklearn

datos = pd.read_csv('../../../datos/iris.csv', delimiter = ';', decimal = ".")
print(datos.info())


datos.iloc[:, 0:4] = StandardScaler().fit_transform(datos.iloc[:, 0:4])
datos

#Debemos crear una instancia de la clase MLPClassifier. Luego podemos definir los siguientes parámetros:

#hidden_layer_sizes: Define la cantidad de capas ocultas y de nodos.

#activation: Define la función de activación a utilizar. Estos pueden ser: identity, logistic, tanh y relu 
#(por defecto).

#solver: Define el solucionador para la optimización de los pesos a utilizar. Estos pueden ser: lbfgs, sgd y #adam (por defecto).

#max_iter: En ocasiones un modelo de redes neuronales puede no converger, esto puede ser causado a que #necesita un mayor número de iteraciones para el solucionador (número de epochs). Por defecto el valor es de #200.


instancia_nnet = MLPClassifier(
  # 3 capas ocultas, la primera con 80 nodos, la segunda con 40 nodos
  # y la tercera con 20 nodos.
hidden_layer_sizes = (80, 40, 20),
  
  # Función de activación relu
activation = "relu",
  
  # Solucionador adam
solver = "adam",
  
  # 300 iteraciones máximas
max_iter = 300,
  
random_state = 0)

analisis_Iris = Analisis_Predictivo(datos, predecir = "tipo", modelo = instancia_nnet, 
                                    train_size = 0.7, random_state = 0)
analisis_Iris.fit_predict()

resultados = analisis_Iris.fit_predict_resultados()

#ejemplotensorflow

datos = pd.read_csv('../../../datos/iris.csv', delimiter = ';', decimal = ".")
print(datos.info())

datos.iloc[:, 0:4] = StandardScaler().fit_transform(datos.iloc[:, 0:4])
datos

from sklearn.preprocessing import LabelEncoder

# Dividimos los datos en variables predictoras y variable a predecir
X = np.array(datos.iloc[:, 0:4])
y = datos["tipo"].ravel()

# Recodificamos la variable a predecir a enteros de 0 a num_clases-1
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y
encoder.classes_

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 0)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Instanciamos nuestra base. Se puede ver como una caja vacía donde vamos añadiendo elementos.
modelo = Sequential()

# Capa de entrada y primera capa oculta
modelo.add(Dense(units = 80, activation = 'relu'))

# Segunda Capa oculta
modelo.add(Dense(units = 40, activation = 'relu'))

# Tercera Capa oculta
modelo.add(Dense(units = 20, activation = 'relu'))
           
# Capa de salida.
# units = 3 debido a que son 3 clases a predecir
modelo.add(Dense(units = 3, activation = 'sigmoid'))

# Configuramos el modelo
# Para los parámetros 
modelo.compile(
  optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
  metrics = 'accuracy')

loss, accu = modelo.evaluate(x_test, y_test, verbose = 0)
accu

pred = modelo.predict(x_test, verbose = 0)
pred = np.argmax(pred, axis = -1)
pred

pred = encoder.inverse_transform(pred)
pred

y_test_classes = encoder.inverse_transform(y_test)

MC = confusion_matrix(y_test_classes, pred, labels = encoder.classes_)
indices = analisis_Iris.indices_general(MC, list(encoder.classes_))
for k in indices:
  print("\n%s:\n%s"%(k,str(indices[k])))



#ejemplo1

datos = pd.read_csv('../../../datos/MuestraCredito5000V2.csv', delimiter = ';', decimal = ".")


datos['IngresoNeto'] = datos['IngresoNeto'].astype('category')
datos['CoefCreditoAvaluo'] = datos['CoefCreditoAvaluo'].astype('category')
datos['MontoCuota'] = datos['MontoCuota'].astype('category')
datos['GradoAcademico'] = datos['GradoAcademico'].astype('category')

datos["MontoCuota"] = datos["MontoCuota"].cat.codes
datos["GradoAcademico"] = datos["GradoAcademico"].cat.codes

datos['MontoCuota'] = datos['MontoCuota'].astype('category')
datos['GradoAcademico'] = datos['GradoAcademico'].astype('category')

datos.head()
datos.info()

#poderpredictivo

analisis_scoring = Analisis_Predictivo(datos, predecir = "BuenPagador", 
                                      train_size = 0.8, random_state = 0)
analisis_scoring.distribucion_variable_predecir()
plt.show()


for var in datos.columns[0:9]:
  if(datos[var].dtype in ['float64', 'int', 'float']):
    analisis_scoring.poder_predictivo_numerica(var)
    plt.show()
  else:
    analisis_scoring.poder_predictivo_categorica(var)
    plt.show()

datos.iloc[:, 0:5] = StandardScaler().fit_transform(datos.iloc[:, 0:5])
datos.head()

#sklearn

nnet = MLPClassifier(hidden_layer_sizes= (50, 25) , max_iter = 8000, 
                     activation = "relu", solver = "adam", random_state = 0)
analisis_scoring = Analisis_Predictivo(datos, predecir = "BuenPagador", modelo = nnet,
                                       train_size = 0.75, random_state = 0)
resultados = analisis_scoring.fit_predict_resultados()

#tensorflow
from sklearn.preprocessing import LabelEncoder

X = np.array(datos.iloc[:, 0:5])
y = datos["BuenPagador"].ravel()
encoder = LabelEncoder()
y = encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state = 0)

modelo = Sequential()
modelo.add(Dense(units = 50, activation = 'relu'))
modelo.add(Dense(units = 25, activation = 'relu'))
modelo.add(Dense(units = 1, activation = 'sigmoid'))
modelo.compile(
  optimizer = 'adam', loss = 'binary_crossentropy',
  metrics = 'accuracy')
  
modelo.fit(x_train, y_train, epochs = 100, batch_size = 16, verbose = 0)

pred = modelo.predict(x_test, verbose = 0)
pred = [1 if p > 0.5 else 0 for p in pred]
pred = encoder.inverse_transform(pred)

y_test_classes = encoder.inverse_transform(y_test)

MC = confusion_matrix(y_test_classes, pred, labels = encoder.classes_)
indices = analisis_scoring.indices_general(MC, list(encoder.classes_))
for k in indices:
  print("\n%s:\n%s"%(k,str(indices[k])))

#ejemplo2
datos = pd.read_csv('../../../datos/SAheart.csv', delimiter = ';', decimal = ".")

datos['famhist'] = datos['famhist'].astype('category')
datos["famhist"] = datos["famhist"].cat.codes
datos['famhist'] = datos['famhist'].astype('category')

datos.head()
datos.info()

analisis_Sheart = Analisis_Predictivo(datos, predecir = "chd", 
                                      train_size = 0.8, random_state = 0)

analisis_Sheart.distribucion_variable_predecir()
plt.show()

for var in datos.columns[0:9]:
  if(datos[var].dtype in ['float64', 'int', 'float']):
    analisis_Sheart.poder_predictivo_numerica(var)
    plt.show()
  else:
    analisis_Sheart.poder_predictivo_categorica(var)
    plt.show()

datos.loc[:, datos.columns != 'chd'] = StandardScaler().fit_transform(datos.loc[:, datos.columns != 'chd'])
datos.head()
nnet = MLPClassifier(hidden_layer_sizes = (5, 10, 10, 5) , max_iter = 50000, 
                     activation = "relu", solver = "adam", random_state = 0)
analisis_Sheart = Analisis_Predictivo(datos, predecir = "chd", modelo = nnet,
                                      train_size = 0.8, random_state = 0)
resultados = analisis_Sheart.fit_predict_resultados()
nnet = MLPClassifier(hidden_layer_sizes = (5, 10, 10, 5), max_iter = 50000, 
                     activation = "tanh", solver = "adam", random_state = 0)
analisis_Sheart = Analisis_Predictivo(datos, predecir = "chd", modelo = nnet,
                                          train_size = 0.8, random_state = 0)
resultados = analisis_Sheart.fit_predict_resultados()
nnet = MLPClassifier(hidden_layer_sizes = (5, 10, 10, 5), max_iter = 50000, 
                     activation = "tanh", solver = "lbfgs", random_state = 0)
analisis_Sheart = Analisis_Predictivo(datos, predecir = "chd", modelo = nnet,
                                          train_size = 0.8, random_state = 0)
resultados = analisis_Sheart.fit_predict_resultados()

#tensorflow
from sklearn.preprocessing import LabelEncoder

X = np.array(datos.iloc[:, 0:9])
y = datos["chd"].ravel()
encoder = LabelEncoder()
y = encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 0)

modelo = Sequential()
modelo.add(Dense(units = 5, activation = 'relu'))
modelo.add(Dense(units = 10, activation = 'relu'))
modelo.add(Dense(units = 10, activation = 'relu'))
modelo.add(Dense(units = 5, activation = 'relu'))
modelo.add(Dense(units = 1, activation = 'sigmoid'))
modelo.compile(
  optimizer = 'adam', loss = 'binary_crossentropy',
  metrics = 'accuracy')
  
modelo.fit(x_train, y_train, epochs = 100, batch_size = 16, verbose = 0)

pred = modelo.predict(x_test, verbose = 0)
pred = [1 if p > 0.5 else 0 for p in pred]
pred = encoder.inverse_transform(pred)

y_test_classes = encoder.inverse_transform(y_test)

MC = confusion_matrix(y_test_classes, pred, labels = encoder.classes_)
indices = analisis_Sheart.indices_general(MC, list(encoder.classes_))
for k in indices:
  print("\n%s:\n%s"%(k,str(indices[k])))

#con otra funcion de activacion tanh

modelo = Sequential()
modelo.add(Dense(units = 5, activation = 'tanh'))
modelo.add(Dense(units = 10, activation = 'tanh'))
modelo.add(Dense(units = 10, activation = 'tanh'))
modelo.add(Dense(units = 5, activation = 'tanh'))
modelo.add(Dense(units = 1, activation = 'sigmoid'))
modelo.compile(
  optimizer = 'adam', loss = 'binary_crossentropy',
  metrics = 'accuracy')
  
modelo.fit(x_train, y_train, epochs = 100, batch_size = 16, verbose = 0)

pred = modelo.predict(x_test, verbose = 0)
pred = [1 if p > 0.5 else 0 for p in pred]
pred = encoder.inverse_transform(pred)

y_test_classes = encoder.inverse_transform(y_test)

MC = confusion_matrix(y_test_classes, pred, labels = encoder.classes_)
indices = analisis_Sheart.indices_general(MC, list(encoder.classes_))
for k in indices:
  print("\n%s:\n%s"%(k,str(indices[k])))


#con otro optimizador 

modelo = Sequential()
modelo.add(Dense(units = 5, activation = 'relu'))
modelo.add(Dense(units = 10, activation = 'relu'))
modelo.add(Dense(units = 10, activation = 'relu'))
modelo.add(Dense(units = 5, activation = 'relu'))
modelo.add(Dense(units = 1, activation = 'sigmoid'))
modelo.compile(
  optimizer = 'sgd', loss = 'binary_crossentropy',
  metrics = 'accuracy')
  
modelo.fit(x_train, y_train, epochs = 100, batch_size = 16, verbose = 0)

pred = modelo.predict(x_test, verbose = 0)
pred = [1 if p > 0.5 else 0 for p in pred]
pred = encoder.inverse_transform(pred)

y_test_classes = encoder.inverse_transform(y_test)

MC = confusion_matrix(y_test_classes, pred, labels = encoder.classes_)
indices = analisis_Sheart.indices_general(MC, list(encoder.classes_))
for k in indices:
  print("\n%s:\n%s"%(k,str(indices[k])))
  













