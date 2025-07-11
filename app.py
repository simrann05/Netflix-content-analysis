import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set Seaborn style
sns.set(style="darkgrid")
plt.rcParams["figure.figsize"] = (10, 5)

# Title
st.title("🎬 Netflix Movies and TV Shows - EDA Dashboard")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")

    # Clean string columns
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Parse date_added
    df['date_added'] = df['date_added'].astype(str).str.strip()
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

    # Feature engineering
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    df['year'] = pd.to_numeric(df['release_year'], errors='coerce')

    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("📊 Filter Options")
type_filter = st.sidebar.multiselect("Select Type", options=df["type"].unique(), default=df["type"].unique())
year_filter = st.sidebar.slider("Select Release Year", int(df["release_year"].min()), int(df["release_year"].max()), (2010, 2020))

filtered_df = df[
    (df["type"].isin(type_filter)) & 
    (df["release_year"].between(year_filter[0], year_filter[1]))
]

st.subheader("📋 Dataset Overview")
st.write(f"Shape: {filtered_df.shape}")
st.dataframe(filtered_df.head())

# Type Distribution
st.subheader("🎥 Distribution of Content Type")
type_counts = filtered_df["type"].value_counts()
fig1, ax1 = plt.subplots()
sns.barplot(x=type_counts.index, y=type_counts.values, palette="Set2", ax=ax1)
ax1.set_title("Movies vs TV Shows")
st.pyplot(fig1)

# Titles Added Over Time
st.subheader("🕒 Titles Added by Year")
titles_by_year = filtered_df["year_added"].value_counts().sort_index()
fig2, ax2 = plt.subplots()
titles_by_year.plot(kind="bar", color="tomato", ax=ax2)
ax2.set_title("Number of Titles Added Each Year")
ax2.set_xlabel("Year")
ax2.set_ylabel("Titles Added")
st.pyplot(fig2)

# Top Genres
st.subheader("🎭 Top 10 Genres")
genre_series = filtered_df["listed_in"].dropna().str.split(', ')
genre_counts = genre_series.explode().value_counts().head(10)
fig3, ax3 = plt.subplots()
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="coolwarm", ax=ax3)
ax3.set_title("Top 10 Genres on Netflix")
st.pyplot(fig3)

# Top Countries
st.subheader("🌍 Top 10 Countries")
country_series = filtered_df["country"].dropna().str.split(', ')
country_counts = country_series.explode().value_counts().head(10)
fig4, ax4 = plt.subplots()
sns.barplot(x=country_counts.values, y=country_counts.index, palette="viridis", ax=ax4)
ax4.set_title("Top 10 Producing Countries")
st.pyplot(fig4)

# Top Directors
st.subheader("🎬 Top 10 Directors")
director_series = filtered_df["director"].dropna().str.split(', ')
director_counts = director_series.explode().value_counts().head(10)
fig5, ax5 = plt.subplots()
sns.barplot(x=director_counts.values, y=director_counts.index, palette="Set3", ax=ax5)
ax5.set_title("Top 10 Directors")
st.pyplot(fig5)

# WordCloud of Actors
st.subheader("🧑‍🎤 WordCloud: Most Frequent Actors")
cast_series = filtered_df["cast"].dropna().str.split(', ')
cast_flat = cast_series.explode()
text = ' '.join(cast_flat.dropna())
wordcloud = WordCloud(width=800, height=400, background_color="black", colormap="Set2").generate(text)
fig6, ax6 = plt.subplots()
ax6.imshow(wordcloud, interpolation="bilinear")
ax6.axis("off")
st.pyplot(fig6)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")

