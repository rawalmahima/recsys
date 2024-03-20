import streamlit as st
from zipfile import ZipFile
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from scipy.spatial.distance import cosine

def dataset_creation(data_path):
  extraction_dir = 'women-fashion'
  if not os.path.exists(extraction_dir):
      os.makedirs(extraction_dir)

  with ZipFile(data_path, 'r') as zip_ref:
      zip_ref.extractall(extraction_dir)

  image_directory = os.path.join(extraction_dir, 'women-fashion')
  image_paths_list = [file for file in glob.glob(os.path.join(image_directory, '*.*')) if file.endswith(('.jpg', '.png', '.jpeg', 'webp'))]

  base_model = ResNet50(weights='imagenet', include_top=False)
  model = Model(inputs=base_model.input, outputs=base_model.output)

  all_features = []
  all_image_names = []

  for img_path in image_paths_list:
      img = image.load_img(img_path, target_size=(224, 224))
      img_array = image.img_to_array(img)
      img_array_expanded = np.expand_dims(img_array, axis=0)
      preprocessed_img = preprocess_input(img_array_expanded)
      

      features = model.predict(preprocessed_img)
      flattened_features = features.flatten()
      normalized_features = flattened_features / np.linalg.norm(flattened_features)
      features = normalized_features
      all_features.append(features)
      all_image_names.append(os.path.basename(img_path))
      input_features = normalized_features
  return input_features, all_features, all_image_names


def recsys(input_image_path,input_features, all_features, all_image_names):
  top_n = 5
  image_paths=[]
  similarities = [1 - cosine(input_features, other_feature) for other_feature in all_features]
  similar_indices = np.argsort(similarities)[-top_n:]
  similar_indices = [idx for idx in similar_indices if idx != all_image_names.index(input_image_path)]
  for i, idx in enumerate(similar_indices[:top_n], start=1):
      image_path = os.path.join('', all_image_names[idx])
      image_paths.append(image_path)
  return image_paths

def results(input_image_path, input_features, all_features, all_image_names):
  result = recsys(input_image_path, input_features, all_features, all_image_names)
  st.write("Displaying Images:")
  col1, col2, col3, col4, col5 = st.columns(5)
  cols = [col1, col2, col3, col4, col5]
  
  for i, path in enumerate(result):
      with cols[i]:
          path = os.path.join('women-fashion/women-fashion',path)
          image = Image.open(path)
          st.image(image, caption=path, use_column_width=True)

def main():
    st.title("Fashion Item Recommender")

    uploaded_file = st.file_uploader("Upload a zip file", type="zip")
    if uploaded_file:
        data_path = 'uploaded.zip'
        with open(data_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write("Creating Dataset:")
        input_features, all_features, all_image_names =  dataset_creation(data_path)
        st.write("Dataset Created with ",len(all_image_names)," images")
        st.write("Select an image for recommendation:")
        selected_image_path = st.selectbox("Select an image", all_image_names)
        input_image_path = selected_image_path
        results(input_image_path, input_features, all_features, all_image_names)
        # selected_image_path = st.selectbox("Select an image", all_image_names, on_change=lambda new_image_path: update_selected_image(new_image_path))
        
        # def update_selected_image(new_image_path):
        #     global selected_image_path
        #     selected_image_path = new_image_path
        #     results(selected_image_path, input_features, all_features, all_image_names)
        
        # results(selected_image_path, input_features, all_features, all_image_names)


        # result = recsys(input_image_path, input_features, all_features, all_image_names)
        # st.write("Displaying Images:")
        # col1, col2, col3, col4, col5 = st.columns(5)
        # cols = [col1, col2, col3, col4, col5]
        
        # for i, path in enumerate(result):
        #     with cols[i]:
        #         path = os.path.join('women-fashion/women-fashion',path)
        #         image = Image.open(path)
        #         st.image(image, caption=path, use_column_width=True)
        
if __name__ == "__main__":
    main()
