import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho do arquivo
file_path = "C:/Users/pedro/OneDrive/Documents/ssd/dados_transformados_v4.xlsx"

# Carregar os dados
data = pd.read_excel(file_path)

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
# Gráfico 1: Gráficos de Linhas - Evolução por Tipo de Crime
# ==============================================
plt.figure(figsize=(12, 8))
for col in crime_types_columns[1:]:
    plt.plot(df['Ano'], df[col], label=col)

plt.title('Evolução Temporal dos Crimes por Tipo (2011-2019)')
plt.xlabel('Ano')
plt.ylabel('Número de Crimes')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Tipos de Crimes")
plt.tight_layout()
plt.show()

# ==============================================
# Gráfico 2: Barras Empilhadas
# ==============================================
plt.figure(figsize=(12, 8))
bottom_values = [0] * len(df['Ano'])
colors = ['cornflowerblue', 'indianred', 'gold', 'mediumpurple', 'lightgreen', 'salmon', 'darkorange', 'skyblue']

for i, col in enumerate(crime_types_columns[1:]):
    plt.bar(df['Ano'], df[col], bottom=bottom_values, label=col, color=colors[i])
    bottom_values = [bottom_values[j] + df[col].iloc[j] for j in range(len(df['Ano']))]

plt.title('Distribuição de Crimes por Tipo (Barras Empilhadas)')
plt.xlabel('Ano')
plt.ylabel('Total de Crimes')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Tipos de Crimes")
plt.tight_layout()
plt.show()

# ==============================================
# Gráfico 3: Pizza - Percentual de Crimes em 2019
# ==============================================
df_2019 = df[df['Ano'] == 2019].drop(columns=['Ano']).sum()  # Crimes totais de 2019
plt.figure(figsize=(8, 8))
plt.pie(df_2019, labels=df_2019.index, autopct='%1.1f%%', colors=colors)
plt.title('Distribuição Percentual de Crimes em 2019')
plt.show()

# ==============================================
# Gráfico 4: Mapa de Calor (Heatmap)
# ==============================================
plt.figure(figsize=(12, 6))
sns.heatmap(df.drop(columns='Ano').transpose(), annot=True, fmt=".0f", cmap='coolwarm', cbar=True)
plt.title('Mapa de Calor - Crimes por Tipo e Ano')
plt.xlabel('Ano')
plt.ylabel('Tipos de Crimes')
plt.show()

# ==============================================
# Gráfico 5: Gráfico de Área Acumulada
# ==============================================
plt.figure(figsize=(12, 8))
df_area = df.set_index('Ano')
df_area.plot.area(stacked=True, colormap='tab10', figsize=(12, 8))
plt.title('Distribuição Acumulada de Crimes por Tipo')
plt.xlabel('Ano')
plt.ylabel('Número de Crimes')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Tipos de Crimes")
plt.tight_layout()
plt.show()

# ==============================================
# Análise Estatística - Resumo
# ==============================================
crime_summary = df.describe().transpose()
print("Resumo Estatístico:")
print(crime_summary)

# Exportar estatísticas para Excel
crime_summary.to_excel("distribuicao_crimes_estatisticas_completa.xlsx", index=True)
print("Resumo estatístico exportado para 'distribuicao_crimes_estatisticas_completa.xlsx'")
