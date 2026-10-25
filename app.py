
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AgriSmart | Crop Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. PREMIUM DESIGN AND READABLE TEXT
# =====================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5faf3, #edf5eb);
    color: #263b2d;
}

[data-testid="stHeader"] {
    background: rgba(245, 250, 243, 0.95);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Main text */
.main p,
.main label,
.main li,
.main h1,
.main h2,
.main h3,
.main h4 {
    color: #263b2d !important;
}

/* Hero banner */
.hero {
    padding: 32px;
    border-radius: 22px;
    background: linear-gradient(120deg, #123c2b, #23834b);
    margin-bottom: 26px;
    box-shadow: 0 8px 24px rgba(18, 60, 43, 0.12);
}

.hero h1 {
    color: #ffffff !important;
    font-size: 2.35rem;
}

.hero p {
    color: #f0fff2 !important;
    font-size: 17px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #d9e8d9;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #526456 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #17683c !important;
    font-weight: 700 !important;
}

/* Form labels */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p {
    color: #263b2d !important;
    font-weight: 600 !important;
}

/* Inputs */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    color: #263b2d !important;
    -webkit-text-fill-color: #263b2d !important;
    background: #ffffff !important;
}

[data-testid="stNumberInput"] button {
    color: #17683c !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: #17683c !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 44px;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #23834b !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #dce8dc;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #263b2d;
}

/* Expanders and captions */
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #dce8dc;
    border-radius: 12px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] p,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #263b2d !important;
}

/* Data tables */
[data-testid="stDataFrame"] {
    color: #263b2d !important;
}

hr {
    border-color: #dce8dc !important;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD DATASET
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv("Crop_recommendation.csv")
    df.columns = df.columns.str.strip()

    required_columns = [
        "N", "P", "K", "temperature",
        "humidity", "ph", "rainfall", "label"
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing)
        )

    df = df[required_columns].copy()

    for col in required_columns[:-1]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().drop_duplicates()

    if df.empty:
        raise ValueError("No valid dataset records were found.")

    return df


try:
    data = load_data()
except FileNotFoundError:
    st.error(
        "Crop_recommendation.csv was not found. "
        "Place it in the same folder as app.py."
    )
    st.stop()
except (ValueError, pd.errors.ParserError) as error:
    st.error(f"Dataset error: {error}")
    st.stop()


features = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]

X = data[features]
y = data["label"]


# =====================================================
# 4. TRAIN RANDOM FOREST MODEL
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


@st.cache_resource
def train_model(training_x, training_y):
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(training_x, training_y)
    return model


model = train_model(X_train, y_train)

test_predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)


# =====================================================
# 5. SHARED PLOTLY THEME
# =====================================================

def style_chart(fig, height=430):
    """Apply consistent, readable chart colors."""

    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="#263b2d"
        ),
        title_font=dict(
            size=20,
            color="#16452d"
        ),
        margin=dict(
            l=100,
            r=70,
            t=45,
            b=65
        ),
        xaxis=dict(
            title_font=dict(
                color="#263b2d",
                size=14
            ),
            tickfont=dict(
                color="#263b2d",
                size=12
            ),
            gridcolor="#e2eae2",
            zerolinecolor="#b8c9b8",
            showline=True,
            linecolor="#b8c9b8"
        ),
        yaxis=dict(
            title_font=dict(
                color="#263b2d",
                size=14
            ),
            tickfont=dict(
                color="#263b2d",
                size=13
            ),
            gridcolor="#edf2ed",
            automargin=True,
            showline=False
        ),
        legend=dict(
            font=dict(
                color="#263b2d",
                size=12
            ),
            bgcolor="#ffffff"
        )
    )

    return fig


# =====================================================
# 6. SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown(
        "<h1 style='color:#17683c;'>🌿 AgriSmart</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='color:#526456;'>CROP INTELLIGENCE PLATFORM</p>",
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Crop Recommendation",
            "Model Insights",
            "About Project"
        ]
    )

    st.divider()

    st.markdown(
        "<p style='color:#526456;'>"
        "Machine Learning · Random Forest"
        "</p>",
        unsafe_allow_html=True
    )


# =====================================================
# 7. HERO BANNER
# =====================================================

st.markdown("""
<div class="hero">
    <h1>🌱 Smart Crop Recommendation</h1>
    <p>
        Explore crop options based on soil nutrients
        and environmental conditions.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 8. METRIC CARDS
# =====================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Test Accuracy",
    f"{accuracy * 100:.2f}%"
)

col2.metric(
    "Crop Categories",
    f"{y.nunique()}"
)

col3.metric(
    "Dataset Records",
    f"{len(data):,}"
)


# =====================================================
# 9. CROP RECOMMENDATION
# =====================================================

if page == "Crop Recommendation":

    st.header("🌾 Enter Your Farm Conditions")

    st.write(
        "Enter measurements from your soil test "
        "and local weather information."
    )

    with st.form("crop_form"):

        left, right = st.columns(2)

        with left:
            st.subheader("🧪 Soil Nutrients")

            nitrogen = st.number_input(
                "Nitrogen (N)",
                min_value=0.0,
                value=90.0,
                step=1.0
            )

            phosphorus = st.number_input(
                "Phosphorus (P)",
                min_value=0.0,
                value=42.0,
                step=1.0
            )

            potassium = st.number_input(
                "Potassium (K)",
                min_value=0.0,
                value=43.0,
                step=1.0
            )

            ph = st.number_input(
                "Soil pH",
                min_value=0.0,
                max_value=14.0,
                value=6.5,
                step=0.1
            )

        with right:
            st.subheader("🌦️ Climate Conditions")

            temperature = st.number_input(
                "Temperature (°C)",
                value=25.0,
                step=0.5
            )

            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0
            )

            rainfall = st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                value=200.0,
                step=5.0
            )

        submitted = st.form_submit_button(
            "🌱 Find Suitable Crop",
            type="primary",
            use_container_width=True
        )

    if submitted:

        user_data = pd.DataFrame(
            [[
                nitrogen,
                phosphorus,
                potassium,
                temperature,
                humidity,
                ph,
                rainfall
            ]],
            columns=features
        )

        # Compare input values with training data ranges.
        outside_range = []

        for feature in features:
            value = user_data.iloc[0][feature]
            low = X[feature].min()
            high = X[feature].max()

            if value < low or value > high:
                outside_range.append(
                    f"{feature}: training range "
                    f"{low:.2f} to {high:.2f}"
                )

        if outside_range:
            st.warning(
                "Some inputs are outside the ranges represented "
                "in the dataset. Predictions may be less reliable."
            )

            for item in outside_range:
                st.write(f"• {item}")

        # Predict the crop and obtain model scores.
        predictions = model.predict_proba(user_data)[0]

        ranking = pd.DataFrame({
            "Crop": model.classes_,
            "Model Score (%)": predictions * 100
        }).sort_values(
            "Model Score (%)",
            ascending=False
        ).reset_index(drop=True)

        best_crop = ranking.iloc[0]["Crop"]
        best_score = ranking.iloc[0]["Model Score (%)"]

        st.divider()
        st.header("🌿 Recommendation Results")

        result_col, chart_col = st.columns([1, 1.2])

        with result_col:
            st.markdown(
                f"""
                <div style="
                    background:#ffffff;
                    border:1px solid #b9ddbf;
                    padding:25px;
                    border-radius:18px;
                ">
                    <p style="color:#526456;font-size:14px;">
                        TOP PREDICTED CROP
                    </p>
                    <h1 style="color:#17683c;">
                        🌱 {str(best_crop).title()}
                    </h1>
                    <p style="color:#263b2d;">
                        Model score: {best_score:.1f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "The model score is not the probability of a "
                "successful harvest."
            )

        with chart_col:
            st.subheader("Top 3 Crop Predictions")

            top_three = ranking.head(3).sort_values(
                "Model Score (%)",
                ascending=True
            )

            fig = px.bar(
                top_three,
                x="Model Score (%)",
                y="Crop",
                orientation="h",
                text="Model Score (%)",
                color_discrete_sequence=["#23834b"]
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
                textfont=dict(
                    color="#263b2d",
                    size=13
                ),
                cliponaxis=False
            )

            fig.update_layout(
                xaxis_title="Model Score (%)",
                yaxis_title="Crop",
                xaxis=dict(
                    range=[
                        0,
                        min(
                            110,
                            max(10, float(top_three["Model Score (%)"].max()) * 1.2)
                        )
                    ]
                )
            )

            fig = style_chart(fig, height=300)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with st.expander("View all crop predictions"):
            ranking["Model Score (%)"] = ranking[
                "Model Score (%)"
            ].round(2)

            st.dataframe(
                ranking,
                use_container_width=True,
                hide_index=True
            )

        st.info(
            "Before planting, check the crop's local growing season, "
            "water availability, soil-test results and regional "
            "agricultural requirements."
        )


# =====================================================
# 10. MODEL INSIGHTS
# =====================================================

elif page == "Model Insights":

    st.header("📊 Model Performance and Insights")

    st.write(
        "The following results evaluate the model on held-out test data."
    )

    st.metric(
        "Test Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    # Feature importance chart
    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=True
    )

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        color_discrete_sequence=["#23834b"]
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        textfont=dict(
            color="#263b2d",
            size=13
        ),
        cliponaxis=False
    )

    fig.update_layout(
        title="Importance of Each Input Feature",
        xaxis_title="Importance",
        yaxis_title="Input Feature"
    )

    fig = style_chart(fig, height=450)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Classification report
    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.caption(
        "Feature importance indicates how the trained model uses "
        "features. It does not prove that a feature causes a crop "
        "to grow successfully."
    )


# =====================================================
# 11. ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.header("🌿 About AgriSmart")

    st.write(
        "AgriSmart is a Machine Learning mini-project that "
        "predicts crop categories from soil and climate measurements."
    )

    st.subheader("Project Details")

    st.markdown("""
    - **Algorithm:** Random Forest Classifier
    - **Learning type:** Supervised Learning
    - **Task:** Multiclass Classification
    - **Language:** Python
    - **Interface:** Streamlit
    - **Data handling:** Pandas
    - **Visualizations:** Plotly
    """)

    st.subheader("Input Features")

    st.markdown("""
    1. Nitrogen (N)
    2. Phosphorus (P)
    3. Potassium (K)
    4. Temperature
    5. Humidity
    6. Soil pH
    7. Rainfall
    """)

    st.subheader("How It Works")

    st.markdown("""
    1. Load and clean the crop dataset.
    2. Separate input features and crop labels.
    3. Split the dataset into training and testing data.
    4. Train the Random Forest classifier.
    5. Evaluate its performance on test data.
    6. Predict a crop category for new measurements.
    """)

    st.subheader("Future Improvements")

    st.markdown("""
    - Add region-specific agricultural information.
    - Include planting season and irrigation availability.
    - Compare Random Forest with other ML algorithms.
    - Validate recommendations against local farm data.
    - Integrate calibrated soil sensors and weather data.
    """)


# =====================================================
# 12. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div style="text-align:center;padding:10px;">
    <p style="color:#526456;">
        <b>AgriSmart</b> · Smart Agriculture through Machine Learning
    </p>
    <p style="color:#526456;">
        Educational crop classification and decision-support project
    </p>
</div>
""", unsafe_allow_html=True)

