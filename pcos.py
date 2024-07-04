import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv(r"C:\Users\Neelam\Downloads\CLEAN- PCOS SURVEY SPREADSHEET.csv")  # Update the file path as necessary

# Define the target column
target_column = 'Have you been diagnosed with PCOS/PCOD?'

# Select the feature columns (all columns except the target column and blood group)
feature_columns = [
    'Age (in Years)',
    'Weight (in Kg)',
    'Height (in Cm / Feet)',
    'After how many months do you get your periods?\n(select 1- if every month/regular)',
    'Have you gained weight recently?',
    'Do you have excessive body/facial hair growth ?',
    'Are you noticing skin darkening recently?',
    'Do have hair loss/hair thinning/baldness ?',
    'Do you have pimples/acne on your face/jawline ?',
    'Do you eat fast food regularly ?',
    'Do you exercise on a regular basis ?',
    'Do you experience mood swings ?',
    'Are your periods regular ?',
    'How long does your period last ? (in Days)\nexample- 1,2,3,4.....'
]

# Separate features and target
X = df[feature_columns]
y = df[target_column]

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)],
    remainder='passthrough')

# Define a pipeline with preprocessing and training
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Define a function to map 'Yes'/'No' to 1/0
def map_to_binary(choice):
    return 1 if choice == 'Yes' else 0

# Define a function to predict using the model
def predict(data):
    input_data = pd.DataFrame(data, index=[0])
    input_data['Have you gained weight recently?'] = map_to_binary(input_data['Have you gained weight recently?'].iloc[0])
    input_data['Do you have excessive body/facial hair growth ?'] = map_to_binary(input_data['Do you have excessive body/facial hair growth ?'].iloc[0])
    input_data['Are you noticing skin darkening recently?'] = map_to_binary(input_data['Are you noticing skin darkening recently?'].iloc[0])
    input_data['Do have hair loss/hair thinning/baldness ?'] = map_to_binary(input_data['Do have hair loss/hair thinning/baldness ?'].iloc[0])
    input_data['Do you have pimples/acne on your face/jawline ?'] = map_to_binary(input_data['Do you have pimples/acne on your face/jawline ?'].iloc[0])
    input_data['Do you eat fast food regularly ?'] = map_to_binary(input_data['Do you eat fast food regularly ?'].iloc[0])
    input_data['Do you exercise on a regular basis ?'] = map_to_binary(input_data['Do you exercise on a regular basis ?'].iloc[0])
    input_data['Do you experience mood swings ?'] = map_to_binary(input_data['Do you experience mood swings ?'].iloc[0])
    input_data['Are your periods regular ?'] = map_to_binary(input_data['Are your periods regular ?'].iloc[0])
    input_data['How long does your period last ? (in Days)\nexample- 1,2,3,4.....'] = input_data['How long does your period last ? (in Days)\nexample- 1,2,3,4.....'].astype(int)
    
    prediction = pipeline.predict(input_data)[0]
    return prediction

# Function to create a timetable for a month
def create_timetable():
    diet_exercise_timetable = {
        'Week 1': {
            'Diet': 'High fiber foods, lean proteins, avoid processed sugar.',
            'Exercise': '30 minutes of brisk walking, 15 minutes of yoga.'
        },
        'Week 2': {
            'Diet': 'Include more vegetables, fruits, whole grains.',
            'Exercise': '20 minutes of cardio, 20 minutes of strength training.'
        },
        'Week 3': {
            'Diet': 'Maintain a balanced diet with good fats like nuts and seeds.',
            'Exercise': '30 minutes of swimming or cycling, 15 minutes of meditation.'
        },
        'Week 4': {
            'Diet': 'Focus on hydration, include herbal teas.',
            'Exercise': '30 minutes of aerobics, 20 minutes of pilates.'
        }
    }
    return diet_exercise_timetable

# Function to get nearby hospitals/clinics
def get_nearby_hospitals():
    # Simulated nearby hospitals for a default location
    hospitals = [
        {"name": "Apollo Hospitals", "address": "Unit 15, plot no 251, Sainik School Rd, Doordarshan Colony, Gajapati Nagar, Bhubaneswar", "contact": "06746661016"},
        {"name": "Sparsh Hospital", "address": "Saheed Nagar, Bhubaneswar, Odisha 751007", "contact": "06746626666"},
        {"name": "Kalinga Hospital", "address": "Maitri Vihar, NALCO Nagar, Chandrasekharpur, Bhubaneswar", "contact": "18005724000"},
    ]
    
    return hospitals

# Streamlit web interface with custom CSS
def main():
    # Custom CSS to change background and text color
    custom_css = """
        <style>
            body {
                background-color: #f0f0f0; /* Light grey background */
                color: #333; /* Dark grey text */
            }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Set title and description
    st.title('PCOS/PCOD Prediction System')
    st.write('Enter your information below to predict if you have been diagnosed with PCOS/PCOD.')

    # Define tabs
    tab1, tab2, tab3 = st.tabs(["Prediction", "Timetable", "Medical Assistance"])

    # Tab 1: Prediction
    with tab1:
        # Collect user input for prediction
        age = st.number_input('Age (in Years)')
        weight = st.number_input('Weight (in Kg)')
        height = st.number_input('Height (in Cm / Feet)')
        periods_months = st.number_input('After how many months do you get your periods? (Numeric)')
        weight_gain = st.radio('Have you gained weight recently?', ('No', 'Yes'))
        excessive_hair = st.radio('Do you have excessive body/facial hair growth?', ('No', 'Yes'))
        skin_darkening = st.radio('Are you noticing skin darkening recently?', ('No', 'Yes'))
        hair_loss = st.radio('Do have hair loss/hair thinning/baldness?', ('No', 'Yes'))
        acne = st.radio('Do you have pimples/acne on your face/jawline?', ('No', 'Yes'))
        fast_food = st.radio('Do you eat fast food regularly?', ('No', 'Yes'))
        exercise = st.radio('Do you exercise on a regular basis?', ('No', 'Yes'))
        mood_swings = st.radio('Do you experience mood swings?', ('No', 'Yes'))
        regular_periods = st.radio('Are your periods regular?', ('No', 'Yes'))
        period_duration = st.number_input('How long does your period last? (in Days)')

        # Prepare input data for prediction
        input_data = {
            'Age (in Years)': age,
            'Weight (in Kg)': weight,
            'Height (in Cm / Feet)': height,
            'After how many months do you get your periods?\n(select 1- if every month/regular)': periods_months,
            'Have you gained weight recently?': weight_gain,
            'Do you have excessive body/facial hair growth ?': excessive_hair,
            'Are you noticing skin darkening recently?': skin_darkening,
            'Do have hair loss/hair thinning/baldness ?': hair_loss,
            'Do you have pimples/acne on your face/jawline ?': acne,
            'Do you eat fast food regularly ?': fast_food,
            'Do you exercise on a regular basis ?': exercise,
            'Do you experience mood swings ?': mood_swings,
            'Are your periods regular ?': regular_periods,
            'How long does your period last ? (in Days)\nexample- 1,2,3,4.....': period_duration
        }

        # Make prediction
        if st.button('Predict'):
            prediction = predict(input_data)
            if prediction == 1:
                st.write('Prediction Result: You have been diagnosed with PCOS/PCOD.')
                st.session_state['diagnosed'] = True
            else:
                st.write('Prediction Result: You have not been diagnosed with PCOS/PCOD.')
                st.session_state['diagnosed'] = False

    # Tab 2: Timetable
    with tab2:
        st.write('If you have been diagnosed with PCOS/PCOD, follow this healthy diet and exercise plan for a month:')
        timetable = create_timetable()
        for week, details in timetable.items():
            st.write(f"### {week}")
            st.write(f"**Diet:** {details['Diet']}")
            st.write(f"**Exercise:** {details['Exercise']}")

    # Tab 3: Medical Assistance
    if 'diagnosed' in st.session_state and st.session_state['diagnosed']:
        with tab3:
            st.write("Based on your prediction result, it is recommended to have a blood test and ultrasound.")
            st.write("Here are some nearby hospitals and clinics for further assistance:")

            hospitals = get_nearby_hospitals()
            for hospital in hospitals:
                st.write(f"**{hospital['name']}**")
                st.write(f"Address: {hospital['address']}")
                st.write(f"Contact: {hospital['contact']}")
                st.write("")

if __name__ == '__main__':
    main()

    

