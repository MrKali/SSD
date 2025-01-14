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
    # Sidebar Menu with Logo
    # ==============================================
    logo_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcThi1pmYJW5VNiJFVH6VwfNpUcRhAwvJy9O9A&s"

    st.sidebar.image(logo_url, use_column_width=True)  # Adiciona o logotipo com ajuste de largura

    menu = st.sidebar.selectbox(
        "Select Analysis Type:",
        ["Main Dashboard", "Data Mining Analysis"]
    )

    # ==============================================
    # Done by (credits)
    # ==============================================
    st.sidebar.markdown("---")  # Horizontal line for separation
    st.sidebar.markdown("**Done by:**")
    st.sidebar.markdown("Fláviu Frisan | Pedro Varela | Gonçalo Dias")

    # ==============================================
    # Main Dashboard
    # ==============================================
    if menu == "Main Dashboard":
        st.title("Main Dashboard")

        # 1. General Crime Distribution
        st.header("1. General Crime Distribution")
        st.subheader("How This Works:")
        st.markdown("""
        This section provides a breakdown of the total number of crimes per municipality for a selected year.
        - **Data Selection:** The user selects a year from the dropdown.
        - **Bar Chart:** Shows the total crimes per municipality for the selected year.
          - **X-axis:** Total number of crimes.
          - **Y-axis:** Municipalities.
        - Interpretation:
          - Longer bars indicate municipalities with higher crime rates.
          - This visualization helps identify crime hotspots in a given year.
        """)

        ano_selecionado = st.selectbox("Select Year:", sorted(df['Ano'].unique(), reverse=True))
        df_ano = df[df['Ano'] == ano_selecionado]
        df_ano_numerico = df_ano.select_dtypes(include='number')
        df_ano_numerico['Municipal'] = df_ano['Municipal']

        total_crimes_municipio = df_ano_numerico.groupby('Municipal').sum(numeric_only=True).sum(axis=1).sort_values(
            ascending=False)
        st.subheader(f"Total Crimes in the Year {ano_selecionado}")
        fig, ax = plt.subplots(figsize=(10, len(total_crimes_municipio) * 0.2))
        total_crimes_municipio.plot(kind='barh', color='cornflowerblue', ax=ax)
        ax.set_title(f"Total Crimes per Municipality in {ano_selecionado}", fontsize=14)
        ax.set_xlabel("Number of Crimes", fontsize=12)
        ax.set_ylabel("Municipalities", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

        # 2. Temporal Crime Comparison
        st.header("2. Temporal Crime Comparison")
        st.subheader("How This Works:")
        st.markdown("""
        This section compares the evolution of crimes over the years for a selected region.
        - **Data Selection:** The user selects a region.
        - **Line Plot:** Shows the trend of crimes over time for the region.
          - **X-axis:** Year.
          - **Y-axis:** Number of crimes.
        - Interpretation:
          - An upward slope indicates an increase in crimes over the years.
          - A downward slope indicates a decrease in crimes.
          - Peaks or drops may indicate events or policy changes that influenced crime rates.
        """)

        regiao_selecionada = st.selectbox("Select Region:", sorted(df['Regiao'].unique()))
        df_regiao = df[df['Regiao'] == regiao_selecionada]
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df_regiao, x='Ano', y='Crimes against patrimony', marker='o', ax=ax1)
        ax1.set_title(f"Crime Trend in the {regiao_selecionada} Region")
        st.pyplot(fig1)

        # 3. Crimes by Region and Municipality
        st.header("3. Crimes by Region and Municipality")
        st.subheader("How This Works:")
        st.markdown("""
        This section highlights the top and bottom 5 municipalities in terms of total crimes.
        - **Data Selection:** Crimes for the selected year.
        - **Tables:**
          - **Top 5 Municipalities:** Shows the municipalities with the highest number of crimes.
          - **Bottom 5 Municipalities:** Shows the municipalities with the lowest number of crimes.
        - Interpretation:
          - Municipalities with high total crimes may need more focused interventions.
          - Municipalities with low crime may indicate effective law enforcement or different socio-economic factors.
        """)

        top_5_maiores = df_ano_numerico.groupby(df_ano['Municipal']).sum(numeric_only=True).sum(axis=1).nlargest(5)
        top_5_menores = df_ano_numerico.groupby(df_ano['Municipal']).sum(numeric_only=True).sum(axis=1).nsmallest(5)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 5 Municipalities with the Most Crimes")
            st.write(top_5_maiores)
        with col2:
            st.subheader("Top 5 Municipalities with the Fewest Crimes")
            st.write(top_5_menores)

        # 4. Crime Distribution by Type
        st.header("4. Crime Distribution by Type")
        st.subheader("How This Works:")
        st.markdown("""
        This section shows how different types of crimes have evolved over time for a selected municipality.
        - **Data Selection:** The user selects a municipality.
        - **Area Chart:** Shows the distribution of crimes over the years.
          - **X-axis:** Year.
          - **Y-axis:** Number of crimes.
        - Interpretation:
          - Areas that grow larger over time indicate an increase in that crime type.
          - The shape of the chart helps identify crime trends for each category.
        """)

        municipio_selecionado = st.selectbox("Select Municipality:", sorted(df['Municipal'].unique()))
        df_municipio = df[df['Municipal'] == municipio_selecionado]
        if not df_municipio.empty:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            df_tipo_crime = df_municipio.drop(columns=['Municipal', 'Regiao']).groupby('Ano').sum()
            df_tipo_crime.plot(kind='area', stacked=True, ax=ax2, colormap='tab10')
            ax2.set_title(f"Crime Types Over Time in {municipio_selecionado}")
            ax2.set_xlabel("Year")
            ax2.set_ylabel("Number of Crimes")
            st.pyplot(fig2)
        else:
            st.warning(f"No data available for the municipality {municipio_selecionado}.")

        # 5. Anomaly Detection
        st.header("5. Anomaly Detection")
        st.markdown("""
        **How This Works:**
        - This section detects **anomalies** in crime data for each municipality.
        - An **anomaly** is identified when the **total number of crimes** in a municipality for a specific year is significantly higher than the usual range.
        - The rule applied for anomaly detection is:

          \[
          \text{Total Crimes} > \text{Mean} + 2 \times \text{Standard Deviation}
          \]

        - **Interpretation:**
          - The **mean** represents the average number of crimes.
          - The **standard deviation** indicates how much the data varies from the average.
          - If the total number of crimes in a municipality for a year exceeds this threshold, it's classified as an anomaly.
        """)

        # Grouping and calculating total crimes per municipality
        df_anomalias = df.groupby(['Ano', 'Municipal']).sum().reset_index()
        df_anomalias['Total Crimes'] = df_anomalias.iloc[:, 3:].sum(axis=1)

        # Calculating mean and standard deviation
        mean_total = df_anomalias['Total Crimes'].mean()
        std_total = df_anomalias['Total Crimes'].std()

        # Setting the anomaly detection threshold
        threshold = mean_total + 2 * std_total

        # Identifying anomalies
        df_picos = df_anomalias[df_anomalias['Total Crimes'] > threshold]

        st.write(f"**Anomaly Detection Threshold:** {threshold:.2f} crimes")
        st.write("Below are the records classified as anomalies:")

        # Displaying the anomalies table with the relevant columns
        st.write(df_picos[['Ano', 'Municipal', 'Total Crimes']])

        # 6. Crime Trends by Type
        st.header("6. Crime Trends by Type")
        st.markdown("""
        **How This Works:**
        - This section shows the trend of a specific crime type across all years.
        - By default, the plot shows "Crimes Against Persons" over time.
        - **Interpretation:**
          - The y-axis represents the number of crimes.
          - The x-axis represents the years (from 2011 to 2019).
          - Peaks in the line indicate years with higher crime rates, possibly linked to specific incidents or regional conditions.
        """)

        # Creating the line plot for "Crimes Against Persons"
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

        # 1. Municipality Clustering Based on Crimes
        st.header("1. Municipality Clustering Based on Crimes")

        # Explanation of how clustering works
        st.subheader("How Clustering Works")
        st.markdown("""
        Municipality clustering groups municipalities based on their crime patterns.  
        The process involves the following steps:
        1. **Data Preparation**: Only numerical crime-related columns are used (e.g., crimes against persons, crimes against property).
        2. **Data Normalization**: Crime values are scaled using standardization (mean 0, standard deviation 1) to ensure that all crime types contribute equally.
        3. **K-Means Clustering**: Municipalities are grouped into clusters based on their similarity in crime rates.
        4. **Interpretation of Clusters**:
           - **Cluster 0**: Typically represents municipalities with low overall crime rates.
           - **Cluster 1**: Represents municipalities with high crime rates.
           - Other clusters may represent municipalities with specific crime patterns (e.g., high property crimes but low crimes against persons).

        Adjust the number of clusters below to see how municipalities are grouped!
        """)

        # Number of clusters slider
        num_clusters = st.slider("Number of Clusters:", min_value=2, max_value=10, value=4)

        # Selecting only numerical columns
        df_numerico = df.select_dtypes(include='number').drop(columns=['Ano'])

        # Normalizing the data
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_numerico), columns=df_numerico.columns)

        # K-Means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df_scaled)

        # Visualization
        st.subheader("Cluster Distribution")
        st.markdown("""
        This chart shows the number of municipalities in each cluster:
        - **X-axis**: Cluster number (e.g., Cluster 0, 1, 2...).
        - **Y-axis**: Number of municipalities in each cluster.
        """)
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.countplot(x='Cluster', data=df, palette='tab10', ax=ax4)
        ax4.set_title("Distribution of Municipalities by Cluster")
        st.pyplot(fig4)

        # Sample of clustered municipalities
        st.subheader("Sample of Municipalities in Each Cluster")
        sample_clusters = df[['Municipal', 'Cluster']].sample(10)
        st.write("Here is a sample of how municipalities were assigned to clusters:")
        st.dataframe(sample_clusters)
        # 2. Crime Prediction by Year
        st.header("2. Crime Prediction by Year")

        # Explanation of how crime prediction works
        st.subheader("How Crime Prediction Works")
        st.markdown("""
        The crime prediction is based on **Linear Regression**, which is a method used to predict future values based on historical data.  
        The process involves the following steps:
        1. **Data Selection**: The user selects a municipality and the types of crimes for which they want to make predictions.
        2. **Feature Selection**: The year (`Ano`) is used as the independent variable (X), and the total number of selected crimes is the dependent variable (y).
        3. **Model Training**: The linear regression model is trained using historical data to learn the trend over time.
        4. **Prediction**: The model predicts the number of crimes for future years (e.g., 2020, 2021, 2022).
        5. **Interpretation**:
           - The **blue points** represent the actual crime data for past years.
           - The **red line** represents the predicted trend for future years.
        """)

        # User input for municipality and crime type selection
        municipio_selecionado = st.selectbox("Select a Municipality for Prediction:", sorted(df['Municipal'].unique()))
        st.subheader("Select Crime Types:")
        crime_options = crime_types_columns[3:]
        selected_crimes = [crime for crime in crime_options if st.checkbox(crime, True)]

        # Check if the user selected at least one crime type
        if not selected_crimes:
            st.warning("Please select at least one type of crime!")
        else:
            # Filter and calculate total crimes for the selected municipality and crime types
            df_municipio = df[df['Municipal'] == municipio_selecionado]
            df_municipio['Total Selected Crimes'] = df_municipio[selected_crimes].sum(axis=1)

            # Variables for the regression model
            X = df_municipio[['Ano']]  # Independent variable (Year)
            y = df_municipio['Total Selected Crimes']  # Dependent variable (Total selected crimes)

            # Train the linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Future predictions
            future_years = pd.DataFrame({'Ano': [2020, 2021, 2022]})
            predictions = model.predict(future_years)

            # Display predictions
            st.subheader("Crime Predictions for Future Years:")
            st.markdown("""
            The table below shows the predicted number of crimes for future years based on the selected municipality and crime types.
            """)
            for year, prediction in zip(future_years['Ano'], predictions):
                st.write(f"Year {year}: **{int(prediction)} crimes predicted**")

            # Plot the actual points and prediction line
            st.subheader("Prediction Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='Ano', y='Total Selected Crimes', data=df_municipio, color='blue', label='Actual Data',
                            ax=ax)
            plt.plot(future_years['Ano'], predictions, color='red', label='Prediction Line')
            plt.title(f"Crime Prediction for {municipio_selecionado}")
            plt.xlabel("Year")
            plt.ylabel("Total Number of Crimes")
            plt.legend()
            st.pyplot(fig)
