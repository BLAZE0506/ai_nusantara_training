import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
#from IPython.display import Image as IPImage, display
import requests
from io import BytesIO
import ast

client= OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

#Step 1: Obtain Landmark COORDINATES for each facial features

def detect_landmarks(image_cv2):  # The function should receive image_cv2 as input
    # Convert the image to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    # Perform face landmarks detection
    result = face_mesh.process(rgb_image)

    # Store landmark coordinates
    landmark_coords = []

    # Draw the face landmarks
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                # Get the coordinates of the landmark
                x = int(landmark.x * image_cv2.shape[1])
                y = int(landmark.y * image_cv2.shape[0])

                # Draw a circle at each landmark point
                cv2.circle(image_cv2, (x, y), 1, (0, 255, 0), -1)

                # Draw the index of the landmark next to the point for verification
                cv2.putText(image_cv2, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    # Extract the face landmarks
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                # Get the coordinates of the landmark
                x = landmark.x
                y = landmark.y
                #z = landmark.z  # Optional, use if needed

                # Append normalized coordinates with index
                landmark_coords.append((idx, x, y))  # Include the index for verification

    return image_cv2, landmark_coords

#Step 2: Masking the Original Image

def mask_facial_features(image_path, landmarks, features_to_mask):
    """
    Mask specified facial features on an image by making them transparent.

    Args:
        image_path (str): Path to the original image.
        landmarks (list): List of (index, x, y) for each facial landmark point (normalized coordinates).
        features_to_mask (dict): Dictionary with keys as feature names and values as lists of landmark indices.

    Returns:
        masked_image (np.ndarray): Image with the specified facial features masked out (transparent).
    """
    # Load the original image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Convert to RGBA if the image is not already
    if image.shape[2] == 3:  # If the image has 3 channels (BGR), convert it to 4 channels (BGRA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    height, width, _ = image.shape

    # Create an alpha mask initialized to fully opaque (255)
    alpha_mask = np.ones((height, width), dtype=np.uint8) * 255

    # Draw polygons for each facial feature
    for feature, indices in features_to_mask.items():
        # Get the points for this feature using the indices
        points = []
        for i in indices:
            if 0 <= i < len(landmarks):
                # Extract the normalized (x, y) coordinates and scale them to the image size
                _, x, y = landmarks[i]
                x = int(x * width)
                y = int(y * height)
                points.append((x, y))

        # Check if points list is not empty
        if len(points) == 0:
            print(f"Warning: No points found for feature '{feature}'. Skipping...")
            continue

        # Convert points to the format required by fillPoly
        points_array = np.array(points, dtype=np.int32)

        # Check the points_array shape
        if points_array.ndim != 2 or points_array.shape[1] != 2:
            print(f"Error: Invalid shape for points in feature '{feature}'. Points array: {points_array}")
            continue

        # Debug print statement to check points
        #print(f"Feature '{feature}': Points = {points_array}")

        # Draw the polygon on the alpha mask (make it transparent by setting the alpha value to 0)
        cv2.fillPoly(alpha_mask, [points_array], 0)

    # Set the alpha channel to be the modified alpha mask
    image[:, :, 3] = alpha_mask

    # Optional: Save the image to verify transparency
    #cv2.imwrite('masked_image.png', image)

    return image

#Step 3: Fetch Prompt from user to know what kind of makeup user prefer
#User will describe what kind of makeup style she wants for what purpose/event

#Step 3: Text model to convert the makeup style for purpose/event type into description of how the user should look like
#Description of makeups for each face features


def makeup_descriptor(prompt):       #input is from user

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
              {"role": "system",
              "content": """Create a suitable prompt to describe the look of the lady for feeding into
                            dalle.
                         """
              },

              {"role": "user",
              "content": prompt
              }
         ],
        #max_tokens=5000,  # Adjust the number of tokens based on the expected length of the output
        #temperature=0.7  # Adjust the creativity of the response
    )

    # Extract and return the generated text
    gpt_response = response.choices[0].message.content
    return gpt_response

#Step 4: Image generation using Dalle-2 with the output description from Step 3

def edit_ai(IMAGE_PATH, MASK_PATH, makeup_descriptor_prompt):
  #prompt=""""""
  response=client.images.edit(
      model='dall-e-2',
      image=open(IMAGE_PATH,"rb"),
      mask=open(MASK_PATH,"rb"),
      prompt=makeup_descriptor_prompt,
      size='512x512',#'256x256',
      )

  #display(IPImage(url=response.data[0].url))
  # Save the image
  image_url = response.data[0].url
  img_data = requests.get(image_url).content
  img = Image.open(BytesIO(img_data))

  # Save the image to the specified save_path
  img.save('new_image.png')

#Step 5: Provide the makeup advise/ steps to get that makeup style using vision model

def makeup_advisor(img_path):       #input is ori image
    model=genai.GenerativeModel("gemini-1.5-flash")
    img = Image.open(img_path) #('/content/cat in spacesuit.png')
    response = model.generate_content([
        """You are a professional makeup artist with many years of experience which can analyse
           makeup style of the a person. Based on the analysed makeup style,
           provide step-by-step instructions of the makeup style procedure to achieve the
           analysed makeup style.
        """, img])
    return response.text


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

st.title("Professional Make-up Tutor")

# Create a file uploader for image files
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Convert the PIL image to a NumPy array (OpenCV format)
    image_np = np.array(image)
    
    # Save the image using OpenCV
    cv2.imwrite('Untitled design (1).png', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    
    # Display the image in Streamlit
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # User input for makeup preferences
    user_prompt = st.text_input("Describe your makeup preferences and purpose (e.g., natural, glamorous, etc.)")

    if user_prompt:
        # Step 1: Load Image from Path
        # Replace 'your_image_path.jpg' with the path to your image file
        image_path = 'Untitled design (1).png'
        image = Image.open(image_path)
        # Get the size of the image
        width, height = image.size

        # Step 2: Display the Image
        #display(IPImage(image_path))  # Display the original image

        # Convert the image to a format suitable for OpenCV
        # Convert to numpy array and then to OpenCV format (BGR)
        image_np = np.array(image)
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)                           # This is the input for the detect_landmark function

        #Step1: Obtain Landmark COORDINATES for each facial features
        image_with_landmarks,landmarks = detect_landmarks(image_cv2)
        # Save the image with landmarks temporarily to display it
        output_image_path = 'image_with_landmarks.jpg'
        # Convert the image back to RGB for displaying with PIL
        image_with_landmarks_rgb = cv2.cvtColor(image_with_landmarks, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_image_path, cv2.cvtColor(image_with_landmarks_rgb, cv2.COLOR_RGB2BGR))


        #Step 2: Masking the Original Image
        # 1. Create Mask Input
        features_to_mask = {
            #'left_eye': [107,66,105,63,70,156,35,31,228,229,230,231,232,233,244,55],
            #'right_eye': [336,296,334,293,300,383,265,261,448,449,450,451,452,453,164,285],
            'both_eyes': [9,336,296,334,293,300,383,265,261,448,449,450,451,452,453,464,285,8,55,244,233,232,231,230,229,228,31,35,156,70,63,105,66,107],
            'nose': [2,97,98,64,102,209,217,174,196,197,419,399,437,429,331,294,327,328],
            'mouth': [0,267,269,270,291,375,321,405,314,17,84,181,91,146,61,185,40,39,37],
            'left_cheek': [120,100,142,206,207,187,123,227,34,143,111,117,118,119],
            'right_cheek': [349,348,347,346,340,372,264,447,352,411,427,426,371,329]
        }
        # 2. Make masking image
        image_without_features=mask_facial_features(image_path, landmarks, features_to_mask)
        # Save the image in a format that supports transparency (PNG)
        output_image_path = 'image_without_features.png'  # Change to .png
        cv2.imwrite(output_image_path, image_without_features)

        #Step 3: Fetch Prompt from user to know what kind of makeup user prefer
        #Text model to convert the makeup style for purpose/event type into description of how the user should look like
        dalle_makeup_prompt=makeup_descriptor(user_prompt)

        #Step 4: Image generation using Dalle-2 with the output description from Step 3
        edit_ai(image_path, output_image_path, dalle_makeup_prompt)

        #Step 5: Provide the makeup advise/ steps to get that makeup style using vision model
        makeup_advise=makeup_advisor('new_image.png')

        st.image('new_image.png')
        st.caption("Makeup procedure step by step")
        st.divider()
        st.write(makeup_advise)




    




