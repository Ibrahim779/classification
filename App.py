# Imports
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from PIL import Image
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from typing import List
import io
import numpy as np
import sys

women_categories = ['Dresses_Casual_Women', 'Evening_Dresses_Women',
              'Jackets_Top_Sport_Women','Jens_Bottom_Casual_Women',
              'Jumpsuits_Casual_Women', 'Pants_Bottom_Formal_Women',
              'Shorts_Bottom_Sport_Women', 'Skirts_Bottom_Casual_Women',
              'Skirts_Bottom_Formal_Women', 'Suits_Bottom_Formal_Women',
              'Suits_Sport_Women', 'Tshirts_Top_Sport_Women',
              'Tops_Casual_Women', 'Tops_Top_Formal_Women', 'Trousers_Bottom_Sport_Women']

men_categories = ['Jackets_Top_Casual_Men', 'Jackets_Top_Formal_Men',
              'Jackets_Top_Sport_Men','Shirts_Top_Casual_Men',
              'Shirts_Top_Formal_Men', 'Shorts_Bottom_Casual_Men',
              'Shorts_Bottom_Sport_Men', 'suits_Sport_Men',
              'Sweatshirt_Top_Casual_Men', 'T_Shirts_Top_Casual_Men',
              'T_Shirts_Top_Casual_Men', 'Trousers_Bottom_Casual_Men',
              'Trousers_Bottom_Formal_Men', 'Trousers_Bottom_Sport_Men']             

# Define the FastAPI app
app = FastAPI()

# Define the Response
class Prediction(BaseModel):
  category: str

# Define the /prediction route
@app.post('/prediction/', response_model=Prediction)
async def prediction_route(gender: int,file: UploadFile = File(...)):

# Load the model
  if (gender == 2 ) :
    filepath = './women_classifier/women_classifier.h5'
    model = load_model(filepath, compile = True)
    categories = women_categories
  else :
    filepath = './men_classifier'
    model = load_model(filepath, compile = True)
    categories = men_categories

  # Get the input shape for the model layer
  input_shape = model.layers[0].input_shape  

# Ensure that this is an image
  if file.content_type.startswith('image/') is False:
    raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

  try:
    # Read image contents
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))

    # Resize image to expected input shape
    pil_image = pil_image.resize((input_shape[1], input_shape[2]))

    # Convert from RGBA to RGB *to avoid alpha channels*
    if pil_image.mode == 'RGBA':
      pil_image = pil_image.convert('RGB')

    # Convert image into grayscale *if expected*
    if input_shape[3] and input_shape[3] == 1:
      pil_image = pil_image.convert('L')

    # Convert image into numpy format
    numpy_image = np.array(pil_image).reshape((input_shape[1], input_shape[2], input_shape[3]))

    # Scale data (depending on your model)
    numpy_image = numpy_image / 255

    # Generate prediction
    prediction_array = np.array([numpy_image])
    predictions = model.predict(prediction_array)
    prediction = predictions[0]
    likely_class = np.argmax(prediction)
    category = categories[likely_class]

    return {
      'category': category
    }
  except:
    e = sys.exc_info()[1]
    raise HTTPException(status_code=500, detail=str(e))
    
# import nest_asyncio
# nest_asyncio.apply()

# if __name__ == '__main__':
#      uvicorn.run(app, host='127.0.0.1', port=8000)