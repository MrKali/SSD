import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ==============================================
# Configuração do Streamlit
# ==============================================
st.set_page_config(page_title="Dashboard de Análise de Crimes", layout="wide")

# ==============================================
# Função para carregar os dados
# ==============================================
@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

# Caminho do arquivo
file_path = "C:/Users/pedro/OneDrive/Documents/ssd/dados_transformados_v4.xlsx"
data = load_data(file_path)

# ==============================================
# Filtrar colunas relevantes
# ==============================================
crime_types_columns = [
    'Ano', 'Municipal', 'Regiao', 'Crimes against persons',
    'Crimes against patrimony', 'Crimes against life in society',
    'Crimes against the State', 'Crimes against pet animals', 'Crimes set out in sundry legislation'
]
df = data[crime_types_columns]

# ==============================================
# Conversão para valores numéricos e filtro
# ==============================================
for col in crime_types_columns[3:]:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# ==============================================
# Menu Lateral
# ==============================================
menu = st.sidebar.selectbox(
    "Selecione a Análise:",
    ["Dashboard Principal", "Análise de Data Mining"]
)

# ==============================================
# 1. Dashboard Principal
# ==============================================
if menu == "Dashboard Principal":
    st.title("Dashboard Principal")

    # 1. Distribuição Geral de Crimes
    st.header("1. Distribuição Geral de Crimes")
    ano_selecionado = st.selectbox("Selecione o Ano:", sorted(df['Ano'].unique(), reverse=True))
    df_ano = df[df['Ano'] == ano_selecionado]
    df_ano_numerico = df_ano.select_dtypes(include='number')
    df_ano_numerico['Municipal'] = df_ano['Municipal']

    total_crimes_municipio = df_ano_numerico.groupby('Municipal').sum(numeric_only=True).sum(axis=1).sort_values(ascending=False)
    st.subheader(f"Total de Crimes no Ano {ano_selecionado}")
    fig, ax = plt.subplots(figsize=(10, len(total_crimes_municipio) * 0.2))
    total_crimes_municipio.plot(kind='barh', color='cornflowerblue', ax=ax)
    ax.set_title(f"Total de Crimes por Município em {ano_selecionado}", fontsize=14)
    ax.set_xlabel("Quantidade de Crimes", fontsize=12)
    ax.set_ylabel("Municípios", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    # 2. Comparação Temporal de Crimes
    st.header("2. Comparação Temporal de Crimes")
    regiao_selecionada = st.selectbox("Selecione a Região:", sorted(df['Regiao'].unique()))
    df_regiao = df[df['Regiao'] == regiao_selecionada]
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df_regiao, x='Ano', y='Crimes against patrimony', marker='o', ax=ax1)
    ax1.set_title(f"Evolução dos Crimes Contra Patrimônio na Região {regiao_selecionada}")
    st.pyplot(fig1)

    # 3. Crimes por Região e Município
    st.header("3. Crimes por Região e Município")
    top_5_maiores = df_ano_numerico.groupby(df_ano['Municipal']).sum(numeric_only=True).sum(axis=1).nlargest(5)
    top_5_menores = df_ano_numerico.groupby(df_ano['Municipal']).sum(numeric_only=True).sum(axis=1).nsmallest(5)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 Municípios com Mais Crimes")
        st.write(top_5_maiores)
    with col2:
        st.subheader("Top 5 Municípios com Menos Crimes")
        st.write(top_5_menores)

    # 4. Distribuição por Tipo de Crime
    st.header("4. Distribuição por Tipo de Crime")
    municipio_selecionado = st.selectbox("Selecione um Município:", sorted(df['Municipal'].unique()))
    df_municipio = df[df['Municipal'] == municipio_selecionado]
    if not df_municipio.empty:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        df_tipo_crime = df_municipio.drop(columns=['Municipal', 'Regiao']).groupby('Ano').sum()
        df_tipo_crime.plot(kind='area', stacked=True, ax=ax2, colormap='tab10')
        ax2.set_title(f"Evolução de Crimes por Tipo no Município {municipio_selecionado}")
        ax2.set_xlabel("Ano")
        ax2.set_ylabel("Número de Crimes")
        st.pyplot(fig2)
    else:
        st.warning(f"Não há dados disponíveis para o município {municipio_selecionado}.")

    # 5. Detecção de Anomalias
    st.header("5. Detecção de Anomalias")
    df_anomalias = df.groupby(['Ano', 'Municipal']).sum().reset_index()
    df_anomalias_numerico = df_anomalias.select_dtypes(include='number')
    desvio_padrao = df_anomalias_numerico.std().mean()
    media_total = df_anomalias_numerico.mean().mean()
    st.write(f"**Desvio-Padrão Médio dos Crimes:** {desvio_padrao:.2f}")
    st.write(f"**Média Geral dos Crimes:** {media_total:.2f}")
    df_picos = df_anomalias[(df_anomalias_numerico.sum(axis=1) > media_total + 2 * desvio_padrao)]
    st.write("Eventos de Pico (Crimes Anormais):")
    st.write(df_picos[['Ano', 'Municipal']])

    # 6. Tendência por Tipo de Crime
    st.header("6. Tendência por Tipo de Crime")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df.groupby('Ano').sum().reset_index(), x='Ano', y='Crimes against persons', marker='o', ax=ax3)
    ax3.set_title("Tendência Temporal de Crimes Contra Pessoas")
    ax3.set_xlabel("Ano")
    ax3.set_ylabel("Número de Crimes")
    st.pyplot(fig3)

# ==============================================
# 2. Análise de Data Mining
# ==============================================
elif menu == "Análise de Data Mining":
    st.title("Análise de Data Mining")

    # 1. Clusterização de Municípios
    st.header("1. Clusterização de Municípios com Base nos Crimes")
    num_clusters = st.slider("Número de Clusters:", min_value=2, max_value=10, value=4)
    df_numerico = df.select_dtypes(include='number').drop(columns=['Ano'])
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numerico), columns=df_numerico.columns)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Cluster', data=df, palette='tab10', ax=ax4)
    ax4.set_title("Distribuição de Municípios por Cluster")
    st.pyplot(fig4)

    # 2. Previsão de Crimes com Regressão Linear
    st.header("2. Previsão de Crimes por Ano")
    municipio_selecionado = st.selectbox("Município para Previsão:", sorted(df['Municipal'].unique()))
    df_municipio = df[df['Municipal'] == municipio_selecionado]  # CORREÇÃO: variável df_municipio definida aqui
    st.subheader("Selecione os Tipos de Crimes:")
    selected_crimes = [crime for crime in crime_types_columns[3:] if st.checkbox(crime, True)]
    if not selected_crimes:
        st.warning("Selecione pelo menos um tipo de crime!")
    else:
        df_municipio['Total Crimes Selecionados'] = df_municipio[selected_crimes].sum(axis=1)
        X = df_municipio[['Ano']]
        y = df_municipio['Total Crimes Selecionados']
        model = LinearRegression()
        model.fit(X, y)
        anos_futuros = pd.DataFrame({'Ano': [2020, 2021, 2022]})
        predicoes = model.predict(anos_futuros)
        st.subheader("Previsão de Crimes para os Próximos Anos:")
        for ano, pred in zip(anos_futuros['Ano'], predicoes):
            st.write(f"Ano {ano}: **{int(pred)} crimes previstos**")

        fig5, ax5 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Ano', y='Total Crimes Selecionados', data=df_municipio, color='blue', label='Pontos Reais', ax=ax5)
        plt.plot(anos_futuros['Ano'], predicoes, color='red', label='Linha de Previsão')
        plt.title(f"Previsão de Crimes Selecionados no Município {municipio_selecionado}")
        plt.xlabel("Ano")
        plt.ylabel("Total de Crimes")
        plt.legend()
        st.pyplot(fig5)

    # 3. Classificação de Municípios com Base nos Crimes
    st.header("3. Classificação de Municípios com Base nos Crimes")
    crime_threshold = st.slider("Defina o Limiar de Criminalidade:", min_value=100, max_value=2000, value=500)
    df_classificacao = df.copy()
    df_classificacao['Total Crimes'] = df_classificacao.iloc[:, 3:].sum(axis=1)
    df_classificacao['Crime Category'] = df_classificacao['Total Crimes'].apply(
        lambda x: 'Alta Criminalidade' if x > crime_threshold else 'Baixa Criminalidade'
    )
    X_class = df_classificacao[['Ano', 'Crimes against persons', 'Crimes against patrimony', 'Crimes against life in society']]
    y_class = df_classificacao['Crime Category']
    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    st.subheader("Matriz de Confusão e Relatório de Classificação")
    st.text("Matriz de Confusão:")
    st.write(confusion_matrix(y_test, y_pred))
    st.text("Relatório de Classificação:")
    st.text(classification_report(y_test, y_pred))
    st.subheader("Exemplo de Classificação")
    amostra = X_test.copy()
    amostra['Classificação Real'] = y_test.values
    amostra['Classificação Prevista'] = y_pred
    st.write(amostra.sample(5))

    # Visualização da Árvore de Decisão
    st.subheader("Visualização da Árvore de Decisão")
    fig6, ax6 = plt.subplots(figsize=(15, 8))
    from sklearn import tree
    tree.plot_tree(classifier, filled=True, feature_names=X_class.columns, class_names=classifier.classes_, ax=ax6)
    st.pyplot(fig6)
