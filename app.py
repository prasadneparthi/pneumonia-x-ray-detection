import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from src.model import build_model
import torch.nn.functional as F

@st.cache_resource
def load_model():
    model = build_model()
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image and the model will predict whether it's Normal or Pneumonia.")

file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probs, 1)

    class_names = ["Normal", "Pneumonia"]
    st.markdown(f"### Prediction: `{class_names[prediction]}`")
    st.markdown(f"### Confidence: `{confidence.item()*100:.2f}%`")
