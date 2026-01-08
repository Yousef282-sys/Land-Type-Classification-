import streamlit as st 
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pandas as pd


# Model architecture

class EuroSAT_CNN(nn.Module):
    def __init__(self, num_classes=13):
        super(EuroSAT_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop1 = nn.Dropout(0.25)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop2 = nn.Dropout(0.25)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.drop3 = nn.Dropout(0.25)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.drop4 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(self.leaky_relu(self.bn2(self.conv2(x))))
        x = self.drop1(self.pool(self.leaky_relu(self.bn3(self.conv3(x)))))
        x = self.drop2(self.pool(self.leaky_relu(self.bn4(self.conv4(x)))))
        x = self.drop3(self.pool(self.leaky_relu(self.bn5(self.conv5(x)))))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.drop4(self.leaky_relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Class names
eurosat_classes = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]
egypt_classes = ["Crop Data", "Desert Data", "Urban Data"]
ALL_CLASSES = eurosat_classes + egypt_classes

# App config
st.set_page_config(page_title="Egypt Land Classifier", layout="wide", page_icon="🌍")
st.title("Egypt Land Classifier")

# Sidebar: navigation
option = st.sidebar.selectbox(
    "Choose Section",
    [
        "Project Overview",
        "Dataset Architecture",
        "Model Architecture",
        "Training Configuration",
        "Performance Metrics",
        "⚡ Testing / Upload",
        "Future Improvements"
    ]
)

# Sidebar: model settings
st.sidebar.header("Model Settings")
MODEL_PATH = st.sidebar.text_input("Model filename", value="finetuned_egypt_model.pth")
use_gpu = st.sidebar.checkbox("Use GPU if available", value=False)

# Load model utility
@st.cache_resource(show_spinner=False)
def load_model(path, device, num_classes=13):
    model = EuroSAT_CNN(num_classes=num_classes)
    try:
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at: {path}")
    model.to(device)
    model.eval()
    return model

# Preprocessing
IMG_SIZE = 128
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Predict function
def predict_image(image: Image.Image, model, device):
    img = image.convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = int(np.argmax(probs))
    return top_idx, float(probs[top_idx]), probs

# Sections
if option == "Project Overview":
    st.subheader("Problem\n")
    st.markdown("""
                    Egypt faces growing challenges in managing its natural resources,
                    agricultural expansion, and urban development due to the lack of
                    accurate and up-to-date data on land use and land cover (LULC)
                    patterns. Traditional field-based mapping methods are time-consuming,
                    expensive, and limited in spatial coverage, making them inefficient for
                    continuous monitoring across the country.\n\n
                """)
    st.subheader("Solution")
    st.markdown("""
                    This project proposes developing an AI-powered land classification
                    model based on Convolutional Neural Networks (CNN). The model will
                    first be trained on the EuroSAT dataset, which contains labeled
                    Sentinel-2 satellite images of Europe. After achieving satisfactory base
                    performance, the model will be fine-tuned with Sentinel-2 images
                    collected from Egypt to specialize it for the Egyptian landscape. This
                    approach combines pre-trained knowledge with localized learning,
                    enabling accurate classification of Egyptian land types.
                """)
elif option == "Dataset Architecture":
    st.header("Dataset Architecture")

    st.markdown("""
    - **EuroSAT Dataset:** Sentinel-2 satellite images covering 10 land-cover classes.
    """)

    #Insert first image (EuroSAT)
    st.image("App Images/eurosat.png", caption="EuroSAT Dataset Classes Split", use_container_width=True)

    st.markdown("""
    - **Egypt Dataset:** Custom land imagery including Crop Data, Desert Data, and Urban Data.
    - All images were resized to **128 × 128** during preprocessing.
    """)

    #Insert second image (Egypt Dataset)
    st.image("App Images/egyptsat.png", caption="Egypt Dataset Classes Split", use_container_width=True)


elif option == "Model Architecture":
    st.header("Model Architecture")
    # Insert architecture image
    st.image("App Images/architecture.png", caption="Model Architecture Diagram", use_container_width=True)

    # Table information
    table_data = {
    "Block": [
        "Block 1",
        "Block 2",
        "Block 3",
        "Block 4",
        "Block 5",
        "Classification Head"
    ],
    "Layers": [
        "Conv2D (3 → 32), BatchNorm, LeakyReLU, MaxPool",
        "Conv2D (32 → 64), BatchNorm, LeakyReLU, MaxPool",
        "Conv2D (64 → 128), BatchNorm, LeakyReLU, MaxPool, Dropout (0.25)",
        "Conv2D (128 → 256), BatchNorm, LeakyReLU, MaxPool, Dropout (0.25)",
        "Conv2D (256 → 512), BatchNorm, LeakyReLU, MaxPool, Dropout (0.25)",
        "Global Average Pooling → Linear (512 → 256) + LeakyReLU + Dropout (0.4) → Linear (256 → 13)"
    ],
    "Output Features": [
        "32",
        "64",
        "128",
        "256",
        "512",
        "13 classes"
    ],
    "Additional Notes": [
        "Initial feature extraction",
        "Edge and texture patterns",
        "Mid-level semantic features",
        "Higher-level feature abstraction",
        "Deep semantic mapping",
        "Final prediction"
    ]
}


    import pandas as pd
    df = pd.DataFrame(table_data)

    # Display table
    st.table(df)




elif option == "Training Configuration":
    st.header("Training Configuration")
    table_data = { 
                  "Property": 
                            [ "Framework",
                             "Input Resolution",
                             "Batch Size",
                             "Train/Validation Split",
                             "Loss Function",
                             "Optimizer",
                             "Epochs",
                             "Evaluation Metrics",
                             "Class Categories" 
                            ],
                      "Details": 
                            [ "PyTorch",
                             "128 × 128 × 3",
                             "32",
                             "Based on directory structure (train & test)",
                             "Cross-Entropy",
                             "Adam (Learning Rate = 0.0001)",
                             "10 epochs",
                             "Training/Testing Accuracy\nTraining/Testing Loss",
                             "10 EuroSAT classes\n3 Custom Egypt classes" 
                             ] 
                            }
    df = pd.DataFrame(table_data)

    # Display table
    st.table(df)

elif option == "Performance Metrics":
    st.header("Performance Metrics")
    st.markdown("""
    - **Accuracy**: ~ 97.5%
    """)
    
    # Insert architecture image
    st.image("App Images/model performance.png", caption="Model Accuracy", use_container_width=True)

    


elif option == "Future Improvements":
    st.header("Future Improvements")
    st.markdown("""
    Egypt’s deserts contain valuable sand types with major industrial importance. Adding sand-type classification enhances the project’s commercial and scientific impact.
    """)
    st.subheader("Why AI sand classification matters")
    st.markdown("""
                    •	Enables large-scale mapping of sand resources\n
                    •	Reduces geological field surveys\n
                    •	Identifies industrially valuable regions\n
                    •	Supports national development projects\n

                """)
    st.subheader("Integration with Current System")
    st.markdown("""
                To support sand classification, the project can be extended with:
                    •	Additional Sentinel-2 spectral bands (especially NIR & SWIR)\n
                    •	A dedicated sand-type detection module\n
                    •	Field-verified labeled datasets\n

                """)
#GUI
elif option == "⚡ Testing / Upload":
    st.header("⚡ Testing / Upload")
    uploaded = st.file_uploader("Upload image (jpg/png)", type=["jpg", "jpeg", "png"])
    col1, col2 = st.columns([1, 1])

    if uploaded:
        image = Image.open(io.BytesIO(uploaded.read()))
        col1.image(image, caption="Uploaded Image", use_container_width=True)

        # Load model
        device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        try:
            model = load_model(MODEL_PATH, device)
        except FileNotFoundError as e:
            st.error(str(e))
            st.info("Place the model file in the folder or set the correct path.")
            st.stop()

        with st.spinner("Classifying..."):
            idx, conf, probs = predict_image(image, model, device)

        predicted_class = ALL_CLASSES[idx]
        st.success(f"Prediction: **{predicted_class}** — Confidence: **{conf*100:.2f}%**")

        # Probability table
        df = pd.DataFrame({
            "class": ALL_CLASSES,
            "probability": (probs * 100).round(3)
        }).sort_values("probability", ascending=False).reset_index(drop=True)
        col2.subheader("Probabilities")
        col2.dataframe(df, use_container_width=True)

        # Top 6 bar chart
        st.subheader("Top predictions")
        st.bar_chart(df.head(6).set_index("class"))

        if st.checkbox("Show raw probabilities (debug)"):
            st.write(probs)
    else:
        st.info("Upload an image to get a prediction.")
