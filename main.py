# # Importing Necessary Libraries
# import tkinter as tk
# from tkinter import filedialog
# from tkinter import *
# from PIL import Image,ImageTk
# import numpy
# import numpy as np

# # Loading the Model
# from keras.models import load_model
# model=load_model('Age_Sex_Detection.h5')

# # Initializing the GUI
# top=tk.Tk()
# top.geometry('800x600')
# top.title('Detecting a person whether Child or Adult')
# top.configure(background='#CDCDCD')

# # Initializing the Labels (1 for age and 1 for Sex)
# label1=Label(top,background="#CDCDCD",font=('arial',15,"bold"))
# label2=Label(top,background="#CDCDCD",font=('arial',15,'bold'))
# label3=Label(top,background="#CDCDCD",font=('arial',15,'bold'))
# sign_image=Label(top)

# # Definig Detect fuction which detects the age and gender of the person in image using the model
# def Detect(file_path):
#     global label_packed
#     image=Image.open(file_path)
#     image=image.resize((48,48))
#     image=numpy.expand_dims(image,axis=0)
#     image=np.array(image)
#     image=np.delete(image,0,1)
#     image=np.resize(image,(48,48,3))
#     print (image.shape)
#     sex_f=["Male","Female"]
#     image=np.array([image])/255
#     pred=model.predict(image)
#     age=int(np.round(pred[1][0]))
#     sex=int(np.round(pred[0][0]))
#     print("Predicted Age is "+ str(age))
#     cat=""
#     if int(age)<=18:
#         cat="child"
#         print("Child")
#     else:
#         cat="Adult"
#         print("Adult")
#     print("Predicted Gender is "+sex_f[sex])
#     label1.configure(foreground="#011638",text=age)
#     label2.configure(foreground="#011638",text=cat)
#     label3.configure(foreground="#011638",text=sex_f[sex])

# # Defining Show_detect button function
# def show_Detect_button(file_path):
#     Detect_b=Button(top,text="Detect Image",command=lambda: Detect(file_path),padx=10,pady=5)
#     Detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
#     Detect_b.place(relx=0.79,rely=0.46) 

# # Definig Upload Image Function
# def upload_image():
#     try:
#         file_path=filedialog.askopenfilename()
#         uploaded=Image.open(file_path)
#         uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
#         im=ImageTk.PhotoImage(uploaded)

#         sign_image.configure(image=im)
#         sign_image.image=im
#         label1.configure(text='')
#         label2.configure(text='')
#         label3.configure(text='')
#         show_Detect_button(file_path)
#     except:
#         pass

# upload=Button(top,text="Upload an Image",command=upload_image,padx=10,pady=5)
# upload.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
# upload.pack(side='bottom',pady=50)
# sign_image.pack(side='bottom',expand=True)
# label1.pack(side="bottom",expand=True)
# label2.pack(side="bottom",expand=True)
# label3.pack(side="bottom",expand=True)
# heading=Label(top,text="Predicting a person whether Child or Adult",pady=20,font=('arial',20,"bold"))
# heading.configure(background="#CDCDCD",foreground="#364156")
# heading.pack()
# top.mainloop()

import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# Load the model
model = load_model('Age_Sex_Detection.h5')

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
