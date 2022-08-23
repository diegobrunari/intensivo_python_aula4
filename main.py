import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


tabela = pd.read_csv('advertising.csv')
print(tabela)

# print(tabela.info()) -> mostra informações na tabela para tratamento.

# print(tabela.corr()) -> mostra correlação entre vendas e as aplicações em tv/radio/jornal (basicamente o que o grafico vai mostrar)

sns.heatmap(tabela.corr(), cmap="YlGnBu", annot=True) #seaborn foi criado a partir do matplotlib porem com graficos mais visuais e faceis de entender

plt.show() #se usa matplotlib para visualizar o grafico, visto que seaborn foi feito a partir dele.

#dividir a tabela em (x) e (y)
#y -> quem eu quero prever (vendas)
#x -> resto (quem eu vou usar pra fazer a previsão <tv/radio/jornal>)

x = tabela[['TV', 'Radio', 'Jornal']]
y = tabela['Vendas']

#toda I.A precisa dessa divisão pra saber o que vai usar para prever.

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.25)

#criando I.A
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

#treinando I.A 
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

#fazer previsão nos testes
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

#calculas R² (quanto seu modelo chegou próximo da respota correta)
from sklearn.metrics import r2_score

print(r2_score(y_teste, previsao_regressaolinear))
print(r2_score(y_teste, previsao_arvoredecisao))

tabela_auxiliar = pd.DataFrame()
tabela_auxiliar['y_teste'] = y_teste
tabela_auxiliar['arvore decisao'] = previsao_arvoredecisao
tabela_auxiliar['regressao linear'] = previsao_regressaolinear

plt.figure(figsize=(15,6))
sns.lineplot(data=tabela_auxiliar)
plt.show() 

#o melhor modelo foi a arvore de decisao

novos = pd.read_csv('novos.csv')
print(novos)

print(modelo_arvoredecisao.predict(novos))