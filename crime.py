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
# Colunas relevantes
crime_types_columns = [
    'Ano', 'Regiao', 'Crimes against persons', 'Crimes of voluntary manslaughter',
    'Crimes against patrimony', 'Crimes against cultural identity and personal integrity',
    'Crimes against life in society', 'Crimes against the State',
    'Crimes against pet animals', 'Crimes set out in sundry legislation'
]

# Filtrar colunas necessárias
df = data[crime_types_columns]

# ==============================================
# Gráfico 1: Total de Crimes por Região em um Ano Específico
# ==============================================
ano_especifico = 2019  # Alterar o ano se necessário
df_ano = df[df['Ano'] == ano_especifico].groupby('Regiao').sum().reset_index()

plt.figure(figsize=(12, 6))
plt.bar(df_ano['Regiao'], df_ano['Crimes against patrimony'], label='Crimes against Patrimony', color='cornflowerblue')
plt.bar(df_ano['Regiao'], df_ano['Crimes against persons'], label='Crimes against Persons', bottom=df_ano['Crimes against patrimony'], color='indianred')
plt.xticks(rotation=45)
plt.title(f'Total de Crimes por Região em {ano_especifico}')
plt.ylabel('Número de Crimes')
plt.legend()
plt.tight_layout()
plt.show()

# ==============================================
# Gráfico 2: Evolução Temporal de Crimes por Região (Total Geral)
# ==============================================
df_region_time = df.groupby(['Ano', 'Regiao']).sum().reset_index()

plt.figure(figsize=(12, 8))
sns.lineplot(data=df_region_time, x='Ano', y='Crimes against patrimony', hue='Regiao', marker="o")
plt.title('Evolução Temporal de Crimes contra Patrimônio por Região (2011-2019)')
plt.xlabel('Ano')
plt.ylabel('Número de Crimes')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# ==============================================
# Gráfico 3: Mapa de Calor de Crimes por Tipo e Região
# ==============================================
df_heatmap = df.groupby(['Regiao', 'Ano']).sum().reset_index()
pivot_table = pd.pivot_table(df_heatmap, values='Crimes against patrimony', index='Regiao', columns='Ano', aggfunc='sum').fillna(0)

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap='Blues')
plt.title('Mapa de Calor - Crimes contra Patrimônio por Região e Ano')
plt.xlabel('Ano')
plt.ylabel('Região')
plt.show()
