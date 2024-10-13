import streamlit as st
import pandas as pd
import joblib

# Load your model, scaler, and encoders
model = joblib.load('model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
metrics = joblib.load('metrics.pkl')
confusion_matrix = joblib.load('confusion_matrix.pkl')

# Function to predict new record
def pred_new(new_record):
    # Convert the new record to a DataFrame
    new_df = pd.DataFrame([new_record])
    
    for col in label_encoders:
        new_df[col] = label_encoders[col].transform(new_df[col])
    
    prediction = model.predict(new_df)
    
    return prediction

# CSS for styling background image
st.markdown("""
    <style>
    .body {
    background: linear-gradient(135deg, #f06, #fceabb);
    }
    .title {
        font-size: 40px;
        color: #FF6347;
        text-align: center;
        font-weight: bold;
    }
    .subtitle {
        font-size: 20px;
        color: #2E8B57;
        text-align: center;
        font-style: italic;
    }
    .section-title {
        color: #4682B4;
        font-size: 24px;
        margin-top: 20px;
    }
    .button-style {
        background-color: #008CBA;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
    }
    .button-style:hover {
        background-color: #005f73;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app layout
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a Page:", ["Home", "Working of Model", "About", "Contributors"])

if page == "Home":
    # Title and subtitle styling
    st.markdown('<p class="title">🧠 StrokeSniffer</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your health, your responsibility!</p>', unsafe_allow_html=True)

    # User input for new record
    st.markdown('<p class="section-title">Fill in your details below</p>', unsafe_allow_html=True)
    
    gender = st.selectbox("What is your gender?", options=['Please Select', 'Male', 'Female'], index=0)  # No default selected
    age = st.number_input("Please enter your age", min_value=0, max_value=120)  # No default value
    hypertension = st.selectbox("Do you have hypertension?", options=['Please Select', 'Yes', 'No'], index=0)  # No default selected
    heart_disease = st.selectbox("Do you have heart disease?", options=['Please Select', 'Yes', 'No'], index=0)  # No default selected
    ever_married = st.selectbox("Have you ever been married?", options=['Please Select', 'Yes', 'No'], index=0)  # No default selected
    work_type = st.selectbox("What is your work type?", options=['Please Select', 'Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'], index=0)  # No default selected
    Residence_type = st.selectbox("What is your residence type?", options=['Please Select', 'Urban', 'Rural'], index=0)  # No default selected
    avg_glucose_level = st.number_input("Please enter your average glucose level (mg/dL) (Note: The normal range for average glucose levels is between 70 and 140 mg/dL.)", min_value=0.0)  # No default value
    bmi = st.number_input("Please enter your BMI (Note: A normal BMI value for a healthy individual is around 23.)", min_value=0.0,max_value=50.0)  # No default value
    smoking_status = st.selectbox("What is your smoking status?", options=['Please Select', 'formerly smoked', 'never smoked', 'smokes'], index=0)  # No default selected


    # Collecting all inputs into a dictionary and converting Yes/No to 1/0
    new_record = {
        'gender': gender,
        'age': age,
        'hypertension': 1 if hypertension == 'Yes' else 0,
        'heart_disease': 1 if heart_disease == 'Yes' else 0,
        'ever_married': 1 if ever_married == 'Yes' else 0,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }


    # Button to trigger prediction
    if st.button("Predict Stroke", help="Click to predict"):
        # Check for empty fields
        if (gender == 'Please Select' or age == 0 or hypertension == 'Please Select' or heart_disease == 'Please Select' or 
            ever_married == 'Please Select' or work_type == 'Please Select' or Residence_type == 'Please Select' or 
            avg_glucose_level == 0.0 or bmi == 0.0 or smoking_status == 'Please Select'):
            st.warning("Please enter all the information asked before proceeding.")
        else:
            prediction = pred_new(new_record)
            if prediction == 1:
                st.error("The model suggests that you may be at risk for a stroke. It's important not to panic, but we strongly recommend that you consult with a healthcare professional for a thorough evaluation. In the meantime, adopting healthier habits like regular exercise, a balanced diet, managing stress, and avoiding smoking can make a significant difference in reducing risks. Early action can be life-saving—take care of yourself!")
            else:
                st.success("Hurray! The model predicts that you are unlikely to have a stroke. Keep up the great work! To maintain your health, continue incorporating regular exercise, a balanced diet, and healthy habits into your routine. Remember, prevention is key—stay active, eat well, and get regular checkups to ensure a healthy future!")


# Working of Model Page
elif page == "Working of Model":
    st.title("Working of the Stroke Prediction Model 🧠")
    st.write("""
        Our stroke prediction model is designed to assess the risk of stroke by analyzing several key features related 
        to your health and lifestyle. Here’s a breakdown of how it works:

        ### Input Stage: Collecting Your Health Data 📝
        When you enter your details into the app, the model collects information on 10 key features that play an 
        important role in determining your stroke risk...

        ### Prediction Stage: Making Sense of the Data 📊
        Once all the data is collected, our model processes the information by encoding categorical features to convert 
        them into a numerical format that the machine learning model can understand.

        ### Output Stage: Stroke Prediction 🎯
        Finally, the processed data is fed into a machine learning model, which predicts whether you are at risk of experiencing a stroke based on the patterns it has learned from historical medical data.

        ### Model Performance Metrics 📈
        - **Train Score**: {train_score:.2f}
        - **Test Score**: {test_score:.2f}
        - **Precision Score**: {precision_score:.2f}
        - **Recall Score**: {recall_score:.2f}
        - **F1 Score**: {f1_score:.2f}

        ### Confusion Matrix 📊
        The confusion matrix helps us understand how well the model is performing. In this case, the matrix is as follows:
        ```
        [[945,  55],
         [ 65, 935]]
        ```
        - True Positives (TP): 935 (correctly predicted strokes)
        - True Negatives (TN): 945 (correctly predicted no strokes)
        - False Positives (FP): 55 (predicted strokes that did not occur)
        - False Negatives (FN): 65 (missed strokes that did occur)

        A higher number of true positives and true negatives indicates better model performance.

        
        """)

    # Display the confusion matrix as a heatmap
    #sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    #st.pyplot()

# About Page
elif page == "About":
    st.title("About StrokeSniffer 🧠💡")
    st.write("""
        Welcome to **StrokeSniffer**, your personalized health companion! This app is designed to help predict the likelihood
        of experiencing a stroke based on critical health and lifestyle factors.

        Leveraging the power of **machine learning**, StrokeSniffer analyzes your inputs—like age, gender, medical history, and 
        more—to provide a prediction. Our model is trained on real medical data to make accurate, data-driven assessments 
        of stroke risk.

        However, it’s important to remember that StrokeSniffer is a **preventive tool** meant to offer guidance, not a 
        diagnostic conclusion. While it provides valuable insights into your health risks, it is always recommended to 
        consult a healthcare professional for an expert evaluation and advice.

        By combining **AI technology** and **user-friendly design**, we aim to empower individuals to take proactive steps 
        toward maintaining their health and well-being. Early detection and awareness can make all the difference when it 
        comes to stroke prevention.

        Whether you’re looking for peace of mind or a prompt to visit your doctor, StrokeSniffer is here to help you on your 
        health journey. Remember—your health is your greatest asset, and prevention is the best cure.

        Take control of your health, and let StrokeSniffer be your guide!
    """)


# Contributors Page
# Contributors Page
elif page == "Contributors":
    st.title("👨‍💻 Contributor")
    st.write("### Developer:")
    st.markdown(f"""
    **Aditya Sangole** 🎓\n
    UG Student, Dept. Of Artificial Intelligence,\n
    G.H. Raisoni College Of Engineering, Nagpur.\n
    📞 **Phone:** +91 9637629918\n
    ✉️ **Email:** [adityasangole12@gmail.com](mailto:adityasangole12@gmail.com)
    
    🔗 [LinkedIn](https://in.linkedin.com/in/aditya-sangole) | 🐙 [GitHub](https://github.com/AdityaSangole)\n
    
    """)

    


