# import io
# import pickle
# import dotenv
# import rembg
# import dotenv
# import numpy as np
# import PIL.Image
# import PIL.ImageOps
# import tensorflow as tf
# import keras
import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image as keras_image
from ChatBotModel import ChatBotModel
from getClassByIndex import getClassByIndex
import dotenv
import rembg
from tensorflow.keras.preprocessing import image

# from ultralytics import YOLO
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model

# from keras.preprocessing import image as keras_image
# from tensorflow.keras.preprocessing import image
# from datetime import datetime, time, timedelta
# from enum import Enum
# from typing import Literal, Union
# from uuid import UUID

# from fastapi import (
#     Body,
#     FastAPI,
#     Query,
#     Path,
#     Cookie,
#     Header,
#     status,
#     Form,
#     File,s
#     UploadFile,
# )
# from pydantic import BaseModel, Field, HttpUrl, EmailStr
# from starlette.responses import HTMLResponse

# from fastapi.middleware.cors import CORSMiddleware
# from ChatBotModel import ChatBotModel
# from getClassByIndex import getClassByIndex
# import matplotlib.pyplot as plt


# def preprocess(input_image, target_size=(224, 224)):
#     # Use rembg to remove the background
#     output_image = rembg.remove(input_image)

#     # Convert the output image to a NumPy array
#     output_array = image.img_to_array(output_image)

#     # Ensure the array has shape (height, width, 3) (RGB format)
#     output_array_rgb = output_array[:, :, :3]

#     # Resize the output array
#     output_array_resized = image.smart_resize(output_array_rgb, target_size)
#     # output_array_resized = output_array_rgb

#     # Add a batch dimension
#     output_array_resized = np.expand_dims(output_array_resized, axis=0)

#     # Normalize the pixel values
#     output_array_resized /= 255.0
#     return output_array_resized


# img = PIL.Image.open("apple.jpeg")
# plt.imshow(img)
# plt.show()
# img_pr = preprocess(img)
# # predicted_class = getClassByIndex(class_index, name.capitalize())
# image_to_plot = np.squeeze(img_pr, axis=0)
# plt.imshow(image_to_plot)
# plt.show()

model = load_model("models\Tea_model.h5")
# # predictions = model.predict(img_pr)
# # class_index = np.argmax(predictions[0])
# # name = "apple"

# # print(np.max(predictions[0]))
