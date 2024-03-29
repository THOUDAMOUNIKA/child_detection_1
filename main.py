import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
#from keras.models import load_model
#from keras.losses import mean_absolute_error
from keras.losses import MeanAbsoluteError
import keras

@keras.saving.register_keras_serializable()
def custom_function():
    # Your function implementation here
    pass

# Load the model
model = load_model('Age_Sex_Detection.h5')
model.compile(loss=["binary_crossentropy", mean_absolute_error], optimizer="Adam", metrics=["accuracy", mean_absolute_error])
#model.save('Age_Sex_Detection.h5',save_format='h5')

# Function to detect age and gender
def detect(uploaded_file):
    image = Image.open(uploaded_file)
    image = image.resize((48, 48))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = np.delete(image, 0, 1)
    image = np.resize(image, (48, 48, 3))
    image = np.array([image]) / 255
    pred = model.predict(image)
    age = int(np.round(pred[1][0]))
    sex = int(np.round(pred[0][0]))
    sex_f = ["Male", "Female"]
    cat = "child" if age <= 18 else "Adult"
    st.text("Predicted Age is " + str(age))
    st.text("Predicted Gender is " + sex_f[sex])
    st.text("Predicted Category is " + cat)

# Streamlit App
def main():
    st.title('Detecting a person whether Child or Adult')
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Button to trigger detection
        if st.button("Detect Image"):
            detect(uploaded_file)

if __name__ == '__main__':
    main()
