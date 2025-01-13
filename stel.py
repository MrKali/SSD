# Instale as bibliotecas, se necessário:
# pip install streamlit pandas matplotlib seaborn openpyxl

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ==============================================
# Configurações do Streamlit
# ==============================================
st.set_page_config(page_title="Análise de Crimes em Portugal", layout="wide")

# ==============================================
# Função para carregar os dados
# ==============================================
@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

# Caminho do arquivo
file_path = "C:/Users/pedro/OneDrive/Documents/ssd/dados_transformados_v4.xlsx"

# Carregar os dados
data = load_data(file_path)

# ==============================================
# Preparação dos Dados
# ==============================================
crime_types_columns = [
    'Ano', 'Crimes against persons', 'Crimes of voluntary manslaughter',
    'Crimes against patrimony', 'Crimes against cultural identity and personal integrity',
    'Crimes against life in society', 'Crimes against the State',
    'Crimes against pet animals', 'Crimes set out in sundry legislation'
]
df = data[crime_types_columns].groupby('Ano').sum().reset_index()

# ==============================================
# Título e Introdução
# ==============================================
st.title("Análise de Crimes em Portugal (2011-2019)")
st.write("Este dashboard apresenta uma análise detalhada da evolução temporal dos diferentes tipos de crimes em Portugal, com gráficos de linha, barras empilhadas, mapas de calor e resumos estatísticos.")

# ==============================================
# Gráfico 1: Gráfico de Linhas
# ==============================================
st.subheader("Evolução Temporal dos Crimes por Tipo (Gráfico de Linhas)")
fig1, ax1 = plt.subplots(figsize=(12, 8))
for col in crime_types_columns[1:]:
    ax1.plot(df['Ano'], df[col], label=col)
ax1.set_title('Evolução Temporal dos Crimes por Tipo (2011-2019)')
ax1.set_xlabel('Ano')
ax1.set_ylabel('Número de Crimes')
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Tipos de Crimes")
st.pyplot(fig1)

# ==============================================
# Gráfico 2: Barras Empilhadas
# ==============================================
st.subheader("Distribuição de Crimes por Tipo (Barras Empilhadas)")
fig2, ax2 = plt.subplots(figsize=(12, 8))
bottom_values = [0] * len(df['Ano'])
colors = ['cornflowerblue', 'indianred', 'gold', 'mediumpurple', 'lightgreen', 'salmon', 'darkorange', 'skyblue']

for i, col in enumerate(crime_types_columns[1:]):
    ax2.bar(df['Ano'], df[col], bottom=bottom_values, label=col, color=colors[i])
    bottom_values = [bottom_values[j] + df[col].iloc[j] for j in range(len(df['Ano']))]
ax2.set_title('Distribuição de Crimes por Tipo (Barras Empilhadas)')
ax2.set_xlabel('Ano')
ax2.set_ylabel('Total de Crimes')
ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Tipos de Crimes")
st.pyplot(fig2)

# ==============================================
# Gráfico 3: Mapa de Calor
# ==============================================
st.subheader("Mapa de Calor - Crimes por Tipo e Ano")
fig3, ax3 = plt.subplots(figsize=(12, 6))
sns.heatmap(df.drop(columns='Ano').transpose(), annot=True, fmt=".0f", cmap='coolwarm', cbar=True, ax=ax3)
ax3.set_title('Mapa de Calor - Crimes por Tipo e Ano')
ax3.set_xlabel('Ano')
ax3.set_ylabel('Tipos de Crimes')
st.pyplot(fig3)

# ==============================================
# Gráfico 4: Gráfico de Área Acumulada
# ==============================================
st.subheader("Distribuição Acumulada de Crimes por Tipo (Gráfico de Área)")
fig4, ax4 = plt.subplots(figsize=(12, 8))
df_area = df.set_index('Ano')
df_area.plot.area(stacked=True, colormap='tab10', ax=ax4)
ax4.set_title('Distribuição Acumulada de Crimes por Tipo')
ax4.set_xlabel('Ano')
ax4.set_ylabel('Número de Crimes')
ax4.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Tipos de Crimes")
st.pyplot(fig4)

# ==============================================
# Análise Estatística - Resumo
# ==============================================
st.subheader("Resumo Estatístico dos Crimes")
crime_summary = df.describe().transpose()
st.write(crime_summary)

# Botão para exportar o resumo estatístico para Excel
if st.button("Exportar Resumo Estatístico para Excel"):
    crime_summary.to_excel("distribuicao_crimes_estatisticas_completa.xlsx", index=True)
    st.success("Resumo estatístico exportado com sucesso para 'distribuicao_crimes_estatisticas_completa.xlsx'")
