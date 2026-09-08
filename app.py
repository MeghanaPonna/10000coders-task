import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)


# -----------------------------------------
# LOAD DATASET
# -----------------------------------------

df = pd.read_csv(
    r"C:\Users\user\Downloads\archive (2)\WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


# -----------------------------------------
# DATA PREPROCESSING
# -----------------------------------------

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)


# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)


# -----------------------------------------
# ENCODE CATEGORICAL COLUMNS
# -----------------------------------------

label_encoders = {}

for column in df.select_dtypes(include="object").columns:

    le = LabelEncoder()

    df[column] = le.fit_transform(df[column])

    label_encoders[column] = le


# -----------------------------------------
# FEATURES AND TARGET
# -----------------------------------------

X = df.drop("Churn", axis=1)

y = df["Churn"]


# -----------------------------------------
# TRAIN TEST SPLIT
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------------------
# TRAIN MODEL
# -----------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------------------
# TITLE
# -----------------------------------------

st.title("📊 Customer Churn Prediction System")

st.write(
    "Predict whether a customer is likely to leave the company."
)

st.divider()


# -----------------------------------------
# CUSTOMER INPUTS
# -----------------------------------------

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Has Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Has Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )


with col2:

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )


# -----------------------------------------
# PREDICTION
# -----------------------------------------

if st.button("🔍 Predict Churn"):

    # Create input using average values
    input_data = pd.DataFrame(
        [X.mean()]
    )

    # Replace numerical values
    input_data["SeniorCitizen"] = senior_citizen
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges


    # Categorical inputs
    categorical_inputs = {
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method
    }


    # Encode categorical values
    for column, value in categorical_inputs.items():

        if column in label_encoders:

            input_data[column] = label_encoders[column].transform(
                [value]
            )[0]


    # -------------------------------------
    # MODEL PREDICTION
    # -------------------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    churn_probability = probability[1]


    st.divider()

    st.subheader("Prediction Result")


    if prediction == 1:

        st.error("⚠️ Customer is likely to churn!")

        st.write(
            f"Churn Probability: **{churn_probability * 100:.2f}%**"
        )

    else:

        st.success("✅ Customer is likely to stay!")

        stay_probability = probability[0]

        st.write(
            f"Retention Probability: **{stay_probability * 100:.2f}%**"
        )


    # -------------------------------------
    # RISK LEVEL
    # -------------------------------------

    st.subheader("Customer Churn Risk")

    st.progress(
        int(churn_probability * 100)
    )


    if churn_probability < 0.30:

        st.success("🟢 Low Churn Risk")

    elif churn_probability < 0.70:

        st.warning("🟡 Medium Churn Risk")

    else:

        st.error("🔴 High Churn Risk")


# -----------------------------------------
# FOOTER
# -----------------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction using Machine Learning | "
    "Python • Scikit-learn • Streamlit"
)