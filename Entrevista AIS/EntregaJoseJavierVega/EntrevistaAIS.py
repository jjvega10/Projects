#!/usr/bin/env python
# coding: utf-8

# # Prueba técnica

# In[70]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## 1. LECTURA Y GUARDADO DE BASES

# In[3]:


clientes_01 = pd.read_csv('CLIENTES_01.csv', sep=';', header=0)
clientes_02 = pd.read_csv('CLIENTES_02.csv', sep=';', header=0)


# In[4]:


clientes_01 = clientes_01.set_index('ID_PERSONA')
clientes_02 = clientes_02.set_index('ID_PERSONA')


# In[5]:


clientes_03 = clientes_01.join(clientes_02, on='ID_PERSONA')
clientes_03.to_csv("CLIENTES03.csv", sep=';')


# ## 2. CONSTRUCCIÓN Y ANÁLISIS DE VARIABLES

# In[6]:


clientes_03['RAT_AP'] = (clientes_03['PAS_CIRCULANTE'] + clientes_03['PAS_LARGO_PLAZO'])/clientes_03['CAPITAL']


# In[7]:


clientes_03.info()


# In[8]:


clientes_03.describe()


# In[9]:


categoricas = ['SECTOR_AIS', 'MORA']
clientes_03_cat = clientes_03.copy()
clientes_03_cat[categoricas] = clientes_03[categoricas].astype('category')


# In[10]:


clientes_03['DOCS_X_COBRAR'][clientes_03['DOCS_X_COBRAR'].notnull()].describe()


# In[11]:


clientes_03[clientes_03['MORA'].isnull()]


# In[12]:


clientes_03[clientes_03['DOCS_X_COBRAR'].isnull()]


# In[13]:


correlaciones = clientes_03.corr()['MORA'].abs().sort_values(ascending=False)
correlaciones


# In[ ]:





# In[14]:


variables_interes = [x for x in correlaciones.drop('MORA').index if correlaciones[x] > 0.10]


# In[71]:


sns.heatmap(clientes_03.corr())
plt.savefig('Correlaciones.png')


# In[16]:


clientes_03[clientes_03['ACTIVO_TOTAL']>0]


# In[72]:


clientes_03[variables_interes].hist(bins=50, figsize=(20,15))
plt.savefig('Histogramas.png')


# In[18]:


correlaciones


# In[19]:


sns.histplot(data=clientes_03_cat, x='COSTO_VTAS', hue='MORA')


# In[25]:


sns.histplot(data=clientes_03_cat, x='VTAS_NETAS_TOTALES', hue='MORA')


# ## MODELIZACIÓN

# ### Limpeza

# In[44]:


clientes_03[clientes_03['MORA'].isnull()]


# In[60]:


clientes_03_modelo = clientes_03_cat.copy()
clientes_03_modelo = clientes_03_modelo[clientes_03_modelo['MORA'].notnull()]
clientes_03_modelo = clientes_03_modelo[clientes_03_modelo['CAPITAL'].notnull()]


# In[61]:


mediana_doc_x_cobrar = clientes_03_modelo['DOCS_X_COBRAR'].median()


# In[62]:


clientes_03_modelo['DOCS_X_COBRAR'] = clientes_03_modelo['DOCS_X_COBRAR'].fillna(mediana_doc_x_cobrar)


# In[63]:


X = clientes_03_modelo.drop(columns='MORA')
y = clientes_03_modelo['MORA']


# In[65]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[66]:


from sklearn.linear_model import LogisticRegression
logit_reg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
logit_reg.fit(X_train, y_train)


# In[73]:


from sklearn.metrics import confusion_matrix
predicciones = logit_reg.predict(X_test)
conf_mat = confusion_matrix(y_test, predicciones)
print('Precision: ', conf_mat[0, 0] / sum(conf_mat[:, 0]))
print('Recall: ', conf_mat[0, 0] / sum(conf_mat[0, :]))
print('Specificity', conf_mat[1, 1] / sum(conf_mat[1, :]))

