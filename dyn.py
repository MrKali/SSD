import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

# ==============================================
# Streamlit Configuration
# ==============================================
st.set_page_config(page_title="Crime Analysis Dashboard", layout="wide")


# ==============================================
# Function to Load Data
# ==============================================
@st.cache_data
def load_data(uploaded_file):
    data = pd.read_excel(uploaded_file)
    return data


# ==============================================
# File Upload Section
# ==============================================
st.title("Crime Analysis Dashboard")
uploaded_file = st.file_uploader("Upload the Excel file with crime data (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    data = load_data(uploaded_file)

    # ==============================================
    # Filter Relevant Columns
    # ==============================================
    crime_types_columns = [
        'Ano', 'Municipal', 'Regiao', 'Crimes against persons',
        'Crimes against patrimony', 'Crimes against life in society',
        'Crimes against the State', 'Crimes against pet animals', 'Crimes set out in sundry legislation'
    ]
    df = data[crime_types_columns]

    # ==============================================
    # Convert to Numeric Values and Fill Missing Data
    # ==============================================
    for col in crime_types_columns[3:]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # ==============================================
    # Sidebar Menu
    # ==============================================
    menu = st.sidebar.selectbox(
        "Select Analysis Type:",
        ["Main Dashboard", "Data Mining Analysis"]
    )

    # ==============================================
    # 1. Main Dashboard
    # ==============================================
    if menu == "Main Dashboard":
        st.title("Main Dashboard")

        # 1. General Crime Distribution
        st.header("1. General Crime Distribution")
        selected_year = st.selectbox("Select Year:", sorted(df['Ano'].unique(), reverse=True))
        df_year = df[df['Ano'] == selected_year]
        df_year_numeric = df_year.select_dtypes(include='number')
        df_year_numeric['Municipal'] = df_year['Municipal']

        total_crimes_by_municipality = df_year_numeric.groupby('Municipal').sum(numeric_only=True).sum(
            axis=1).sort_values(ascending=False)
        st.subheader(f"Total Crimes in {selected_year}")
        fig, ax = plt.subplots(figsize=(10, len(total_crimes_by_municipality) * 0.2))
        total_crimes_by_municipality.plot(kind='barh', color='cornflowerblue', ax=ax)
        ax.set_title(f"Total Crimes by Municipality in {selected_year}", fontsize=14)
        ax.set_xlabel("Number of Crimes", fontsize=12)
        ax.set_ylabel("Municipalities", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

        # 2. Temporal Crime Comparison
        st.header("2. Temporal Crime Comparison")
        selected_region = st.selectbox("Select Region:", sorted(df['Regiao'].unique()))
        df_region = df[df['Regiao'] == selected_region]
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df_region, x='Ano', y='Crimes against patrimony', marker='o', ax=ax1)
        ax1.set_title(f"Crime Trend Against Patrimony in {selected_region}")
        st.pyplot(fig1)

        # 3. Crimes by Region and Municipality
        st.header("3. Crimes by Region and Municipality")
        top_5_highest = df_year_numeric.groupby(df_year['Municipal']).sum(numeric_only=True).sum(axis=1).nlargest(5)
        top_5_lowest = df_year_numeric.groupby(df_year['Municipal']).sum(numeric_only=True).sum(axis=1).nsmallest(5)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 5 Municipalities with the Most Crimes")
            st.write(top_5_highest)
        with col2:
            st.subheader("Top 5 Municipalities with the Fewest Crimes")
            st.write(top_5_lowest)

        # 4. Crime Distribution by Type
        st.header("4. Crime Distribution by Type")
        selected_municipality = st.selectbox("Select a Municipality:", sorted(df['Municipal'].unique()))
        df_municipality = df[df['Municipal'] == selected_municipality]
        if not df_municipality.empty:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            df_crime_type = df_municipality.drop(columns=['Municipal', 'Regiao']).groupby('Ano').sum()
            df_crime_type.plot(kind='area', stacked=True, ax=ax2, colormap='tab10')
            ax2.set_title(f"Crime Evolution by Type in {selected_municipality}")
            ax2.set_xlabel("Year")
            ax2.set_ylabel("Number of Crimes")
            st.pyplot(fig2)
        else:
            st.warning(f"No data available for the municipality {selected_municipality}.")

        # 5. Anomaly Detection
        st.header("5. Anomaly Detection")
        df_anomalies = df.groupby(['Ano', 'Municipal']).sum().reset_index()
        df_anomalies_numeric = df_anomalies.select_dtypes(include='number')
        std_deviation = df_anomalies_numeric.std().mean()
        mean_total = df_anomalies_numeric.mean().mean()
        st.write(f"**Average Standard Deviation of Crimes:** {std_deviation:.2f}")
        st.write(f"**Overall Crime Mean:** {mean_total:.2f}")
        df_peaks = df_anomalies[(df_anomalies_numeric.sum(axis=1) > mean_total + 2 * std_deviation)]
        st.write("Peak Events (Anomalous Crimes):")
        st.write(df_peaks[['Ano', 'Municipal']])

        # 6. Crime Trend by Type
        st.header("6. Crime Trend by Type")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df.groupby('Ano').sum().reset_index(), x='Ano', y='Crimes against persons', marker='o',
                     ax=ax3)
        ax3.set_title("Temporal Trend of Crimes Against Persons")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Number of Crimes")
        st.pyplot(fig3)

    # ==============================================
    # 2. Data Mining Analysis
    # ==============================================
    elif menu == "Data Mining Analysis":
        st.title("Data Mining Analysis")

        # 1. Municipality Clustering
        st.header("1. Municipality Clustering Based on Crimes")
        num_clusters = st.slider("Select Number of Clusters:", min_value=2, max_value=10, value=4)
        df_numeric = df.select_dtypes(include='number').drop(columns=['Ano'])
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df_scaled)
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.countplot(x='Cluster', data=df, palette='tab10', ax=ax4)
        ax4.set_title("Municipality Distribution by Cluster")
        st.pyplot(fig4)

        # 2. Crime Prediction with Linear Regression
        st.header("2. Crime Prediction by Year")
        selected_municipality = st.selectbox("Municipality for Prediction:", sorted(df['Municipal'].unique()))
        df_municipality = df[df['Municipal'] == selected_municipality]

        # Checkboxes for selecting types of crimes
        st.subheader("Select Crime Types:")
        crime_options = crime_types_columns[3:]
        selected_crimes = [crime for crime in crime_options if st.checkbox(crime, True)]

        if not selected_crimes:
            st.warning("Please select at least one type of crime!")
        else:
            df_municipality['Total Crimes Selected'] = df_municipality[selected_crimes].sum(axis=1)
            X = df_municipality[['Ano']]
            y = df_municipality['Total Crimes Selected']
            model = LinearRegression()
            model.fit(X, y)

            # Predictions for future years
            future_years = pd.DataFrame({'Ano': [2020, 2021, 2022]})
            predictions = model.predict(future_years)

            st.subheader("Crime Predictions for Upcoming Years:")
            for year, pred in zip(future_years['Ano'], predictions):
                st.write(f"Year {year}: **{int(pred)} predicted crimes**")

            fig5, ax5 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='Ano', y='Total Crimes Selected', data=df_municipality, color='blue', label='Actual Data',
                            ax=ax5)
            ax5.plot(future_years['Ano'], predictions, color='red', label='Prediction Line')
            ax5.set_title(f"Crime Prediction in {selected_municipality}")
            ax5.set_xlabel("Year")
            ax5.set_ylabel("Total Crimes")
            ax5.legend()
            st.pyplot(fig5)
