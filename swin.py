import tkinter as tk
from tkinter import Button, Canvas, Scrollbar
from tkinter import font  # Import font module
from PIL import Image, ImageTk
import cv2
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch.nn.functional as F

# Load model and extractor
model = AutoModelForImageClassification.from_pretrained("swinasl")
extractor = AutoFeatureExtractor.from_pretrained("swinasl")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Dictionary for mapping prediction results
id2label = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
            19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del',
            27: 'nothing', 28: 'space'}

# Function to capture image from the blue box and classify when Predict button is pressed
def classify_image():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        return

    # Crop image according to the blue box
    crop_img = frame[50:300, 50:300]

    # Preprocess image for VIT model
    inputs = extractor(images=[Image.fromarray(crop_img)], return_tensors="pt")
    
    # Predict with VIT model
    with torch.no_grad():
        logits = model(**inputs).logits

    # Calculate probabilities using softmax
    probabilities = F.softmax(logits, dim=-1)
    
    # Get the highest probability and its class index
    max_prob, predicted_class_idx = torch.max(probabilities, dim=-1)
    
    # Get the probability value
    max_prob = max_prob.item()
    
    # If the highest probability is less than 0.6, don't perform classification
    if max_prob < 0.2:
        print(f"Low confidence: {max_prob}, not displaying result.")
        return

    # Get predicted label if probability is high enough
    predicted_label = id2label[predicted_class_idx.item()]
    
    # If prediction is 'space', add space to text box
    if predicted_label == 'space':
        text_box.insert(tk.END, ' ')
    elif predicted_label == 'del':
        # If 'del', delete the last character in the text box
        current_text = text_box.get("1.0", tk.END)
        if len(current_text) > 1:  # Check if there's any text to delete
            text_box.delete("end-2c", tk.END)  # Delete the last character
    else:
        # Add result to text box
        text_box.insert(tk.END, predicted_label)

# Function to update the camera feed continuously
def update_frame():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret:
        # Define blue box coordinates
        start_point = (50, 50)
        end_point = (300, 300)

        # Apply Gaussian blur to the entire frame with larger kernel
        blurred_frame = cv2.GaussianBlur(frame, (51, 51), 0)  # Increased blur kernel size

        # Copy the region inside the blue box from the original (non-blurred) frame
        non_blurred_region = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        
        # Replace the blurred region with the non-blurred region
        blurred_frame[start_point[1]:end_point[1], start_point[0]:end_point[0]] = non_blurred_region

        # Draw blue box on the frame
        color = (255, 0, 0)  # Blue color
        thickness = 2  # Thickness of the rectangle border
        cv2.rectangle(blurred_frame, start_point, end_point, color, thickness)

        # Convert frame to ImageTk format
        img = Image.fromarray(cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.imgtk = imgtk
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    # Call the function again after 10 ms to keep the video feed running
    root.after(10, update_frame)

# Function to clear classification results
def clear_text():
    text_box.delete(1.0, tk.END)

# Setup GUI with tkinter
root = tk.Tk()
root.title("ASL Real-time Classifier Using Swin Transformers")

# Set layout to make the camera feed the main focus
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=0)

# Create camera view on canvas
canvas = Canvas(root, width=500, height=500, bg="white")
canvas.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Create text box to display classification results with a scrollbar
text_box_frame = tk.Frame(root)
text_box_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

# Set font size for text box
custom_font = font.Font(family="Helvetica", size=20)  # Increased font size
text_box = tk.Text(text_box_frame, height=5, width=40, font=custom_font)
text_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = Scrollbar(text_box_frame, command=text_box.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

text_box.config(yscrollcommand=scrollbar.set)

# Create button to clear text
clear_button = Button(root, text="CLEAR", command=clear_text)
clear_button.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

# Create button to predict
predict_button = Button(root, text="PREDICT", command=classify_image)
predict_button.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Start updating the camera feed
update_frame()

# Run GUI
root.mainloop()

# Make sure to release camera after program ends
cap.release()
cv2.destroyAllWindows()
