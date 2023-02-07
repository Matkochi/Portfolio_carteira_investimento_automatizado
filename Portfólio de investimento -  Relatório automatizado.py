#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install riskfolio-lib')
get_ipython().system('pip install yfinance')


# In[3]:


#Importação das bibliotecas

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
import riskfolio as rp
warnings.filterwarnings('ignore')


# In[4]:


#In sample


#Período
inicio = '2016-01-01'
fim = '2022-08-30'

#Seleção dos ativos da carteira
ativos = ['PETR4.SA','VALE3.SA', 'BBDC4.SA','ITSA4.SA', 'ABEV3.SA' ,'WEGE3.SA', 'IVVB11.SA']

#Peso da carteira anteriormente

peso_in = np.array([0.15,0.15,0.15,0.15,0.15,0.15,0.10])


# In[5]:


#Download dos dados

carteira = yf.download(ativos, start=inicio, end=fim)['Adj Close']
carteira.head()


# In[6]:


#Extração dos parâmetros

#Retornos
retorno_carteira = carteira.pct_change().dropna()

#Covariância
cov_in = retorno_carteira.cov()


# In[9]:


#Retorno out of sample


out_inicio = '2022-09-01'
out_fim = '2023-01-20'


#Download dos dados
carteira_out = yf.download(ativos, start = out_inicio, end = out_fim)['Adj Close']

#Calculo Retorno
retorno_out = carteira_out.pct_change().dropna()

#Matriz covariancia out-of-sample
cov_out = retorno_out.cov()


display(retorno_out.head())


# In[7]:


pesos_in = pd.DataFrame(data={'pesos_in':peso_in},index=ativos)
pesos_in


# In[26]:


ax = rp.plot_series(returns = retorno_carteira, w=pesos_in,cmap='tab20',height=6,width=10,ax=None)


# In[10]:


ax = rp.plot_dendrogram(returns=retorno_carteira,
                      codependence='pearson',
                      linkage='single',
                      k=None,
                      max_k=10,
                      leaf_order=True,
                      ax=None)


# In[ ]:


##Modelo de otimização

Copyright (c) 2020-2022, Dany Cajas All rights reserved.  

fonte: https://riskfolio-lib.readthedocs.io/en/latest/index.html

Marcos López de Prado. Building diversified portfolios that outperform out of sample. The Journal of Portfolio Management, 42(4):59–69, 2016. URL: https://jpm.pm-research.com/content/42/4/59, arXiv:https://jpm.pm-research.com/content/42/4/59.full.pdf, doi:10.3905/jpm.2016.42.4.059.


# In[12]:


pd.options.display.float_format = '{:.4%}'.format

portfolio = rp.HCPortfolio(returns=retorno_carteira)

model='HRP' 
codependence = 'pearson'
rm = 'MV' 
rf = 0 # 
linkage = 'single' 

leaf_order = True 

pesos = portfolio.optimization(model=model,
                      codependence=codependence,
                      rm=rm,
                      rf=rf,
                      linkage=linkage,
                      leaf_order=leaf_order)
display(pesos)


# In[13]:


#Retorno out of sample


fig_2, ax_2 = plt.subplots(figsize=(1,1))

rp.plot_series(returns=retorno_out, w=pesos, cmap='tab20', height=6, width=10,
                    ax=None)
plt.savefig('cum_ret.png');


# In[14]:


#Gráfico de composição dos novos pesos antes

fig_2, ax_2 = plt.subplots(figsize=(6,2))

rp.plot_pie(w=pesos_in, title='Portfolio', height=6, width=10,
                 cmap="tab20", ax=None)
plt.savefig('pf_weights_in.png');


# In[15]:


#Gráfico de composição dos novos pesos da carteira otimizada

fig_3, ax_3 = plt.subplots(figsize=(6,2))

rp.plot_pie(w=pesos, title='Portfolio', height=6, width=10,
                 cmap="tab20", ax=None)
plt.savefig('pf_weights_out.png');


# In[16]:


#Parametros do portfolio otimizado

media_retorno = portfolio.mu
covariancia = portfolio.cov
retornos = portfolio.returns


# In[17]:


#Gráfico de contribuição de medida de risco por ativo carteira as is

fig_4, ax_4 = plt.subplots(figsize=(6,2))

rp.plot_risk_con(w=pesos_in, cov=cov_in, returns=retorno_carteira, rm=rm,
                      rf=0, alpha=0.05, color="tab:blue", height=6,
                      width=10, t_factor=252, ax=None)
plt.savefig('risk_cont_in.png');


# In[18]:


#Gráfico de contribuição de medida de risco por ativo carteira as is

fig_5, ax_5 = plt.subplots(figsize=(6,2))

rp.plot_risk_con(w=pesos, cov=cov_out, returns=retorno_out, rm=rm,
                      rf=0, alpha=0.05, color="tab:blue", height=6,
                      width=10, t_factor=252, ax=None)
plt.savefig('risk_cont_out.png');


# In[19]:


##Histograma de retornors do portfólio
fig_6, ax_6 = plt.subplots()

rp.plot_hist(returns=retorno_carteira, w=pesos_in, alpha=0.05, bins=50, height=6,
                  width=10, ax=None)
plt.savefig('pf_returns_in.png');


# In[20]:


#Histograma dos retornos do portfolio

fig_7, ax_7 = plt.subplots()

rp.plot_hist(returns=retorno_out, w=pesos, alpha=0.05, bins=50, height=6,
                  width=10, ax=None);
plt.savefig('pf_returns_out.png')


# In[21]:


##Tbela de medida de risco

fig_8, ax_8 = plt.subplots(figsize=(6,2))
rp.plot_table(returns=retorno_carteira, w=pesos_in, MAR=0, alpha=0.05, ax=None)
plt.savefig('table_in.png');


# In[22]:


fig_9, ax_9 = plt.subplots(figsize=(6,2))
rp.plot_table(returns=retorno_out, w=pesos, MAR=0, alpha=0.05, ax=None)
plt.savefig('table_out.png');


# In[24]:


##Contruindo o relatório em PDF
get_ipython().system('pip install FPDF')
from fpdf import FPDF


# In[25]:


# 1. Setup básico do PDF

#Criar o pdf

pdf = FPDF()

#Adiciona uma nova página
pdf.add_page()

#Setup da fonte
pdf.set_font('Arial', 'B', 16)

# 2. layout do pdf
pdf.cell(40,10, 'Diagnóstico da sua Carteira')

#Quebra de linha
pdf.ln(20)


# 3. Tabela performance
pdf.cell(20,7, 'Como sua carteira performou de {} até {}'.format(inicio,fim))
pdf.ln(8)
pdf.image('table_in.png', w=180, h=200)
pdf.ln(60)

# 4. Tabela peformance out-of-sample
pdf.cell(20, 7, 'Como sua carteira performou de {} até {}'.format(out_inicio,out_fim))
pdf.ln(8)
pdf.image('table_out.png', w=180, h=200)
pdf.ln(60)

# 5. Retorno Acumulado Carteira
pdf.cell(20, 7, 'Retorno Acumulado da Carteira de {} até {}'.format(out_inicio,out_fim))
pdf.ln(8)
pdf.image('cum_ret.png', w=120, h=70)
pdf.ln(10)

         
# 6. Pesos         
pdf.cell(20, 7, 'Pesos Carteira Atual')
pdf.ln(8)
pdf.image('pf_weights_in.png', w=100, h=60)
pdf.ln(10)
pdf.cell(20, 7, 'Pesos Carteira Otimizada')
pdf.ln(8)
pdf.image('pf_weights_out.png', w=100, h=60)
pdf.ln(30)
         
         
         
# 7. Contribuição de risco por ativo
pdf.cell(20, 7, 'Contribuição de risco por ativo de {} até {}'.format(inicio,fim))
pdf.ln(15)
pdf.image('risk_cont_in.png',w=150, h=80)
pdf.ln(20)
pdf.cell(20, 7, 'Contribuição de risco por ativo de {} até {}'.format(out_inicio,out_fim))
pdf.ln(15)
pdf.image('risk_cont_out.png',w=150, h=80)
pdf.ln(80)         


# 8. Histograma de retornos
pdf.cell(20, 7, 'Histograma de retornos de {} até {}'.format(inicio,fim))
pdf.ln(15)
pdf.image('pf_returns_in.png', w=150, h=80)
pdf.ln(20)

pdf.cell(20, 7, 'Histograma de retornos de {} até {}'.format(out_inicio,out_fim))
pdf.ln(15)
pdf.image('pf_returns_out.png', w=150, h=80)
pdf.ln(20)       
         

# 9. Disclaimer
pdf.set_font('Times', '', 6)
pdf.cell(5, 2, 'Relatório construído com a biblioteca RiskFolio https://riskfolio-lib.readthedocs.io/en/latest/index.html')


# 10. Output do PDF file
pdf.output('diagnostico_de_carteira.pdf', 'F')


# In[ ]:




