import io
import dotenv
import rembg
import dotenv
import numpy as np
import PIL.Image
import PIL.ImageOps
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image as keras_image
from tensorflow import image


from fastapi import (
    FastAPI,
    File,
    UploadFile,
)

from keras.preprocessing import image as keras_image
from getClassByIndex import getClassByIndex

dotenv.load_dotenv()
app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


def preprocess(input_image, target_size=(224, 224)):
    # Use rembg to remove the background
    output_image = rembg.remove(input_image)

    # Convert the output image to a NumPy array
    output_array = keras_image.img_to_array(output_image)

    # Ensure the array has shape (height, width, 3) (RGB format)
    output_array_rgb = output_array[:, :, :3]

    # Resize the output array
    output_array_resized = image.smart_resize(output_array_rgb, target_size)
    # output_array_resized = output_array_rgb

    # Add a batch dimension
    output_array_resized = np.expand_dims(output_array_resized, axis=0)

    # Normalize the pixel values
    output_array_resized /= 255.0
    return output_array_resized


@app.get("/")
async def root():
    return {"hello"}


@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    name = "apple"
    contents = await file.read()
    pil_image = PIL.Image.open(io.BytesIO(contents))
    # print(pil_image)
    # plt.imshow(pil_image)
    # plt.axis("off")
    # plt.show()
    # print("befor", pil_image.size)
    pil_image = preprocess(pil_image)
    # print(pil_image.shape)
    # plt.imshow(pil_image)
    # plt.axis("off")
    # plt.show()
    model = load_model("models\Apple_model.h5")
    predictions = model.predict(pil_image)
    # print(predictions)
    class_index = np.argmax(predictions[0])
    predicted_class = getClassByIndex(class_index, name.capitalize())
    # print(predicted_class)
    return {"predicted class": predicted_class}
