#2.15.0
#2.16.1
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Skin Cancer Prediciton")
## -------------------------------------------------------
# Tab 1
tab1, tab2, tab3, tab4 = st.tabs(["Home", "🗃 Data", "Model", "📈 Chart"])

# Display content in Tab 1
tab1.markdown("""
### About Skin cancer

Skin cancer is the most common form of cancer globally, with millions of new cases diagnosed each year. Early detection is crucial for successful treatment, making automated detection systems valuable tools in the fight against this disease. The HAM10000 dataset, a large collection of skin lesion images, has been used to develop machine learning models for the classification of skin lesions. 

This project aims to create a predictive model using the HAM10000 dataset to classify skin lesions into one of seven categories, including both benign and malignant conditions.

By using machine learning techniques and convolutional neural networks (CNNs), the project aims to develop a predictive model capable of accurately classifying skin lesions into these categories. The HAM10000 dataset provides a rich source of data for this purpose, and successful model development could contribute to improved detection and diagnosis of skin cancer and other skin conditions.

You can find the source code in the [GitHub Repository](https://github.com/aman9650/Skin_cancer_detection)
""")



### ------------------------------------------------------------
# Tab 2

# Display content in Tab 2
tab2.markdown("""
# Dataset

The HAM10000 dataset contains a variety of skin lesion images, including seven common classes:

1. **Melanoma**: The most dangerous form of skin cancer.
2. **Melanocytic nevus**: Common, benign moles or "beauty marks."
3. **Basal cell carcinoma**: The most common type of skin cancer, often appearing as a small, shiny bump.
4. **Actinic keratosis**: Rough, scaly patches that can develop into skin cancer.
5. **Benign keratosis**: Non-cancerous growths that may resemble skin cancer.
6. **Dermatofibroma**: Harmless, small, and firm bumps that may appear red or brown.
7. **Vascular lesion**: Benign skin lesions that include various types of blood vessel abnormalities.
""")


# Load data
df = pd.read_csv("clean_data.csv")
# Create an expander within Tab 2
see_data=tab2.expander('You can click here to see the raw data first 👉')
# With the expander context, display dataframe
with see_data:
    st.dataframe(data=df.reset_index(drop=True))

## ------------------------------------------------------

# Tab3
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('final_model .h5')

# Define the class labels
class_labels = ['Actinic keratoses (akiec)', 'Basal cell carcinoma (bcc)', 'Benign keratosis-like lesions (bkl)',
                'Dermatofibroma (df)', 'Melanoma (mel)', 'Melanocytic nevi (nv)', 'Vascular lesions (vasc)']

def preprocess_image(image):
    image = image.resize((65, 65))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict(image):
    image = preprocess_image(image)
    predictions = model.predict(image)
    return predictions[0]

# Define a dictionary mapping class labels to cancer information and symptoms
cancer_info = {
    'class_label_1': {
        'name': 'Actinic Keratoses (AKIEC)',
        'info': 'Actinic keratoses (AK), also known as solar keratoses, are rough, scaly patches on the skin caused by excessive sun exposure.',
        'symptoms': ['Rough, scaly patches on the skin', 'Itching or burning in the affected area', 'Red or pink coloration']
    },
    'class_label_2': {
        'name': 'Basal Cell Carcinoma (BCC)',
        'info': 'Basal cell carcinoma (BCC) is a type of skin cancer that begins in the basal cells of the skin.',
        'symptoms': ['Raised, pink or red, pearly-looking bump', 'Open sore that oozes or bleeds, then crusts and heals', 'Scar-like area that is white, yellow, or waxy']
    },
    'class_label_3': {
        'name': 'Benign Keratosis-Like Lesions (BKL)',
        'info': 'Benign keratosis-like lesions (BKL) are non-cancerous skin growths that may resemble other skin conditions.',
        'symptoms': ['Irregular, scaly, or crusty patches', 'Slowly growing or changing growths', 'Variety of colors including red, pink, brown, or black']
    },
    'class_label_4': {
        'name': 'Dermatofibroma (DF)',
        'info': 'Dermatofibroma (DF) is a non-cancerous skin growth that often appears as a small, firm bump on the skin.',
        'symptoms': ['Firm, round, pink or brown bump', 'Dimple-like depression when pinched', 'Itching or tenderness']
    },
    'class_label_5': {
        'name': 'Melanoma (MEL)',
        'info': 'Melanoma (MEL) is a type of skin cancer that develops from melanocytes, the cells that produce melanin.',
        'symptoms': ['New or changing mole', 'Irregular borders or uneven color', 'Itching, bleeding, or crusting']
    },
    'class_label_6': {
        'name': 'Melanocytic Nevi (NV)',
        'info': 'Melanocytic nevi (NV), commonly known as moles, are benign growths on the skin composed of melanocytes.',
        'symptoms': ['Small, round, symmetrical growths', 'Uniform color (usually brown)', 'No significant change in appearance over time']
    },
    'class_label_7': {
        'name': 'Vascular Lesions (VASC)',
        'info': 'Vascular lesions (VASC) are abnormalities in the blood vessels that may appear as birthmarks, vascular malformations, or vascular tumors.',
        'symptoms': ['Red or purple discoloration on the skin', 'Raised or flat lesions', 'Bleeding or ulceration in severe cases']
    }
}

def main():


    # Upload image in tab 3
    uploaded_file = tab3.file_uploader('Upload Dermatoscopic image of skin lesions :', type=['jpg', 'png'])

    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)

        # Display the uploaded image in tab 3
        tab3.image(image, caption='Uploaded Image', use_column_width=True)

        # Make predictions
        predictions = predict(image)

        # Display the predicted class label in tab 3
        predicted_class_index = np.argmax(predictions)
        predicted_class = f'class_label_{predicted_class_index + 1}'
        tab3.write(f'**Predicted Class:** {class_labels[predicted_class_index]}')

        # Display information about the predicted cancer and its symptoms in tab 3
        if predicted_class in cancer_info:
            tab3.write(f'**Information:** {cancer_info[predicted_class]["info"]}')
            tab3.write('**Symptoms:**')
            for symptom in cancer_info[predicted_class]["symptoms"]:
                tab3.write(symptom)
        else:
            tab3.write("Information not available for this predicted cancer type.")

        # Plot percentage of each class found in tab 4

        fig, ax = plt.subplots(figsize=(10, 8))

        bars = ax.bar(class_labels, predictions * 100, color=plt.cm.viridis(np.linspace(0, 1, len(class_labels))))
        ax.set_xlabel('Class')
        ax.set_ylabel('Percentage')
        ax.set_title('Percentage of Each Class')
        ax.set_xticks([])
        ax.grid(True)

        legend_labels = [f'{label}: {pred*100:.2f}%' for label, pred in zip(class_labels, predictions)]
        ax.legend(bars, legend_labels, loc='upper right', bbox_to_anchor=(1.3, 1))

        tab4.pyplot(fig)

        # # Plot percentage of each class found
        # plt.figure(figsize=(10, 8))
        # bars = plt.bar(class_labels, predictions * 100, color=plt.cm.viridis(np.linspace(0, 1, len(class_labels))))
        # plt.xlabel('Class')
        # plt.ylabel('Percentage')
        # plt.title('Percentage of Each Class')
        # plt.xticks([])
        # plt.grid(True)
        # legend_labels = [f'{label}: {pred*100:.2f}%' for label, pred in zip(class_labels, predictions)]
        # plt.legend(bars, legend_labels, loc='upper right', bbox_to_anchor=(1.3, 1))
        # st.pyplot(plt)

if __name__ == '__main__':
    main()
