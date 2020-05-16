#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:20:30 2020
@author: gfd
"""
import pandas as pd
import numpy as np
from datapackage import Package
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import r2_score

# FUNÇÃO PARA FILTRAR DATAFRAME
def filtradfPrecosAnp(campo):
    global dfPrecosAnp
    global title
    itemList = dfPrecosAnp[campo].unique().tolist()
    strItens=''
    for s in itemList:
        strItens = strItens + str(itemList.index(s)) + ' - ' + s + '; '
    answer = int(input('Escolha um ' + campo + '(' + strItens + ') ===> '))
    dfPrecosAnp = dfPrecosAnp[(dfPrecosAnp[campo] == itemList[answer])]
    if (itemList[answer]=='GLP'):
        dfPrecosAnp["preco_revenda"] = dfPrecosAnp["preco_revenda"]/13
        dfPrecosAnp["preco_aquisicao"] = dfPrecosAnp["preco_aquisicao"]/13
    title = title + itemList[answer] + '; '

# 2. Como os dados da ANP são semanais, cria lista diária
def expandeSemanalparaDiario(df):
    nro_columns = df.columns.size
    cols = df.columns
    days_ahead = 7
    precos_diarios = np.zeros(shape=(len(df)*days_ahead,nro_columns), dtype='O')
    i=0
    for index, row in df.iterrows():
        #print(row['data'], row['preco_revenda'],row['preco_aquisicao'])
        proxData = row['data']
        for dow in range(days_ahead):

            for l in range(nro_columns):
               # print(cols[l])
                if (cols[l]=="data"):
                    precos_diarios[i][l] = proxData
                else:
                    precos_diarios[i][l] = row[cols[l]]
                    row[cols[l]] = None
                    
            i+=1
            proxData = (pd.Timestamp(proxData) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    return precos_diarios

def avaliaModelo(title, regressor, X_test, y_test, y_pred):
    combined = np.vstack((y_test,y_pred,(y_pred/y_test)*100)).T
    #print("Intercepto",regressor.intercept_)
    #print("Coeficiente",regressor.coef_)
    print("Média dos Acertos: ",np.average(combined[:,2]))
    print("Desvio Padrão dos Acertos: ",np.std(combined[:,2]))
    print("Média dos valores reais", np.average(y_test))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R Squared:',r2_score(y_test,y_pred))
    print(["valor real","valor previsto","acerto%"])
    print(combined[:10])
    if (X_test.shape[1]==1):
        plt.title(title + " Resultado do Teste - vermelho=previsto") 
        plt.scatter(X_test, y_test,  color='black')
        plt.plot(X_test, y_pred, color='red', linewidth=3)
        plt.ylabel("valor_y")
        plt.xlabel("valor_x")
        plt.show()
    plt.title(title + " Taxa de Acerto") 
    plt.ylabel("% do acerto")
    plt.xlabel("valor real")
    plt.scatter(combined[:,0],combined[:,2],c='b')
    plt.show()
    return combined

# 1. Consulta preços praticados de combustíveis e cria dataframe - fonte: ANP
anos = ["2018","2019","2020"]
#anos = ["2020"]
title=''
dfPrecosAnp = pd.DataFrame(columns=["data","uf","municipio","produto","preco_revenda","preco_aquisicao"])
for ano in anos:
    url = "http://www.anp.gov.br/images/Precos/Semanal2013/"
    url = url + "SEMANAL_MUNICIPIOS-" + ano +".xlsx"
    #url = "SEMANAL_MUNICIPIOS-" + ano +".xlsx"
    dfTemp = pd.read_excel(url,skiprows=14,nrows=200000,usecols=[0,3,4,5,8,14],na_values=["-"])
    dfTemp.columns=["data","uf","municipio","produto","preco_revenda","preco_aquisicao"]
    #dfPrecosAnp.append(dfTemp, ignore_index=True)
    dfPrecosAnp = pd.concat([dfPrecosAnp,dfTemp],ignore_index=True)

filtradfPrecosAnp("produto")
filtradfPrecosAnp("uf")
filtradfPrecosAnp("municipio")
dfPrecosAnp = dfPrecosAnp.drop(["produto","municipio","uf"],axis=1)
dfPrecosAnp = pd.DataFrame(data=expandeSemanalparaDiario(dfPrecosAnp),columns=["data","preco_revenda","preco_aquisicao"])
dfPrecosAnp["data"] = pd.to_datetime(dfPrecosAnp['data']).dt.date
dfPrecosAnp["preco_revenda"]=pd.to_numeric(dfPrecosAnp["preco_revenda"])
dfPrecosAnp["preco_aquisicao"]=pd.to_numeric(dfPrecosAnp["preco_aquisicao"])
dfPrecosAnp = dfPrecosAnp.fillna(dfPrecosAnp.interpolate()) 

# 6. BUSCA VALORES DO CAMBIO (BRL X USD) NO BANCO CENTRAL
url = "https://ptax.bcb.gov.br/ptax_internet/consultaBoletim.do?method=gerarCSVFechamentoMoedaNoPeriodo&ChkMoeda=61"
url = url + "&DATAINI=01/01/" + anos[0] + "&DATAFIM=31/12/" + anos[len(anos)-1]
dfCambio = pd.read_csv(url, usecols=[0,5], delimiter=";", header=None, decimal=",")
dfCambio.columns = ["data","cambio"]
dfCambio.data =  "0" + dfCambio.data.astype(str)
dfCambio.data = dfCambio.data.str[-8:]
dfCambio.data = pd.to_datetime(dfCambio.data, format="%d%m%Y").dt.date

# 3. Busca dados do preço internacional do petróleo (em USD)
package1 = Package('https://datahub.io/core/oil-prices/datapackage.json')
#package.save("teste.zip")
#package1 = Package('teste.zip')

#4. Cria 2 listas por tipo de petroleo: BRENT e WTI
for resource in package1.resources:
   if resource.descriptor['name']=='brent-daily_csv':
       brentPckg = resource.read()
       size = len(brentPckg)
       brent = np.zeros(shape=(size,2), dtype='O')
       i=0
       for item in brentPckg:
           brent[i] = [item[0],round(item[1],3)]
           i+= 1
       del brentPckg
   if resource.descriptor['name']=='wti-daily_csv':
       wtiPckg = resource.read()
       size = len(wtiPckg)
       wti = np.zeros(shape=(size,2), dtype='O')
       i=0
       for item in wtiPckg:
           wti[i] = [item[0],round(item[1],3)]
           i+= 1
       del wtiPckg

dfBrent = pd.DataFrame(data=brent,columns=["data","brent_usd_bbl"])
dfBrent["brent_usd_bbl"] = pd.to_numeric(dfBrent["brent_usd_bbl"])
dfWti = pd.DataFrame(data=wti,columns=["data","wti_usd_bbl"])
dfWti["wti_usd_bbl"] = pd.to_numeric(dfWti["wti_usd_bbl"])
del brent
del wti
dfPetroleo = dfBrent.merge(dfWti, on="data")
#dfPetroleo["data"] = pd.DatetimeIndex(dfPetroleo['data']) + pd.DateOffset(-7)
dfPetroleo["data"]=pd.to_datetime(dfPetroleo['data']).dt.date
dfPetroleo = dfPetroleo[(dfPetroleo["data"] >= pd.to_datetime(anos[0] + '-01-01'))]
dfPetroleo.plot.line(x="data",y=["brent_usd_bbl","wti_usd_bbl"],title="Preço USD Barril Petróleo")

# CRIA DATAFRAME DO PREÇO DO PETROLEO CONVERTENDO DE USD/BARRIL PARA BRL/LITRO
litrosporBarril = 159;
dfPetroleo["brent_usd_l"]=pd.to_numeric(dfPetroleo["brent_usd_bbl"])/litrosporBarril
#Converte preço do Petróleo Brent (referencia) para BRL
dfPetroleo = dfPetroleo.merge(dfCambio, on='data',how="inner")
dfPetroleo["brent_brl_l"]=pd.to_numeric(dfPetroleo["brent_usd_l"])*dfPetroleo.cambio

##PREÇOS INTERNACIONAIS derivados
litrosporUSG = 3.785
urls_eia = {"gasolina_usd_l":"https://www.eia.gov/petroleum/gasdiesel/xls/pswrgvwall.xls","diesel_usd_l":"https://www.eia.gov/petroleum/gasdiesel/xls/psw18vwall.xls"}
dfPrecosIntl = pd.DataFrame(columns=["data"])
for intlProd in urls_eia:
    print(intlProd,urls_eia[intlProd])
    dfTemp = pd.read_excel(urls_eia[intlProd],sheet_name='Data 1',skiprows=2,nrows=200000,usecols=[0,1])
    dfTemp.columns=["data",intlProd]
    dfTemp = pd.DataFrame(data=expandeSemanalparaDiario(dfTemp),columns=["data",intlProd])
    dfTemp["data"]=pd.to_datetime(dfTemp['data']).dt.date
    dfTemp = dfTemp[(dfTemp["data"] >= pd.to_datetime(anos[0] + '-01-01'))]
    dfTemp[intlProd] = pd.to_numeric(dfTemp[intlProd])/litrosporUSG
    dfPrecosIntl = dfPrecosIntl.merge(dfTemp, on="data",how='right')

#comparacom precos derivados internacionais
dfPrecosIntl = dfPrecosIntl.merge(dfPetroleo[["data","brent_usd_l"]],on='data', how='right')
dfPrecosIntl = dfPrecosIntl.fillna(dfPrecosIntl.interpolate()) 
dfPrecosIntl = dfPrecosIntl.fillna(method='bfill') 
dfPrecosIntl["fator_intl_gasolina"] = dfPrecosIntl["gasolina_usd_l"]/dfPrecosIntl["brent_usd_l"]
dfPrecosIntl["fator_intl_diesel"] = dfPrecosIntl["diesel_usd_l"]/dfPrecosIntl["brent_usd_l"]

# 7. CONSOLIDA DADOS EM ÚNICO DATAFRAME
dfGeral = pd.DataFrame(columns=["data"])
dfGeral = dfGeral.merge(dfPrecosAnp, on="data", how="right")
dfGeral = dfGeral.merge(dfPrecosIntl[["data","fator_intl_gasolina","fator_intl_diesel"]], on='data')
dfGeral = dfGeral.merge(dfPetroleo[["data","brent_brl_l","cambio"]], on='data')
#dfGeral = dfGeral.fillna(dfGeral.interpolate()) #preenche Nan com valores interpolados
dfGeral["fator_nacional"] = dfGeral['preco_revenda']/dfGeral['brent_brl_l']
desc = dfGeral.describe()
corr = dfGeral.corr()
print(desc)
print(corr)

dfPrecosIntl.plot.line(x="data",y=["gasolina_usd_l","diesel_usd_l","brent_usd_l"], title="Preços Mercado EUA")
dfGeral.plot.line(x="data",y=["fator_intl_gasolina","fator_intl_diesel","fator_nacional"],title='Fator ' + title)
dfGeral.plot.line(x="data",y=["preco_aquisicao","preco_revenda","brent_brl_l"], title=title)
dfGeral.plot.scatter(x="brent_brl_l",y="preco_revenda", title=title)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
#from sklearn.tree import DecisionTreeRegressor

# MODELO MERCADO EUA
X = dfPrecosIntl.iloc[:, 3:4].values 
y = dfPrecosIntl.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# EXECUTA TREINAMENTO
regressor = LinearRegression()
#regressor = SVR(kernel = 'rbf')
#regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
# AVALIA MODELO
print("### AVALIAÇÃO MODELO MERCADO EUA ###")
combinedEua = avaliaModelo('EUA1',regressor,X_test,y_test,y_pred)

#MODELO MERCADO BRASILEIRO 1 - X = brent
X = dfGeral.iloc[:, 5:6].values 
y = dfGeral.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# EXECUTA TREINAMENTO
regressor = LinearRegression()
#regressor = SVR(kernel = 'rbf')
#regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
# AVALIA MODELO
print("### AVALIAÇÃO MODELO 1 MERCADO BRASIL ###")
combinedBra1 = avaliaModelo("BR1",regressor,X_test,y_test,y_pred)

#MODELO MERCADO BRASILEIRO 2 - X = brent, cambio
X = dfGeral.iloc[:,[5,6]].values 
y = dfGeral.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# EXECUTA TREINAMENTO
#regressor = LinearRegression()
regressor = SVR(kernel = 'rbf')
#regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
# AVALIA MODELO
print("### AVALIAÇÃO MODELO 2 MERCADO BRASIL ###")
combinedBra2 = avaliaModelo("BR2",regressor,X_test,y_test,y_pred)

