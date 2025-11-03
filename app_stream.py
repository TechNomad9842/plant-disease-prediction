import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import google.generativeai as genai

# --- Your Custom PyTorch Model Architecture ---
class ImageClassificationBase(nn.Module):
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_diseases)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# --- Class Labels and Model Loading ---
class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
                'Tomato___healthy']

@st.cache_resource
def load_model():
    model = CNN_NeuralNet(3, len(class_labels))
    model.load_state_dict(torch.load("plant_disease_params.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# --- Initialize Gemini ---
def initialize_gemini(api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    return model

# --- Image Transformation ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Session State Initialization ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None

# --- Streamlit User Interface ---
st.set_page_config(page_title="Plant Disease Detection & Assistant", page_icon="🌿", layout="wide")

st.title("🌿 Plant Disease Detection & AI Assistant")

# Sidebar for API Key
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password", help="Get your API key from https://makersuite.google.com/app/apikey")
    
    if api_key:
        if st.session_state.gemini_model is None:
            try:
                st.session_state.gemini_model = initialize_gemini(api_key)
                st.success("✅ Gemini API connected!")
            except Exception as e:
                st.error(f"❌ Error connecting to Gemini: {str(e)}")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This app uses AI to detect plant diseases and provides an intelligent chatbot to answer your questions.")

# Create two columns
col1, col2 = st.columns([1, 1])

# --- Left Column: Disease Detection ---
with col1:
    st.header("📸 Disease Detection")
    st.markdown("Upload an image of a plant leaf for disease analysis.")
    
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        if st.button("🔍 Predict Disease", use_container_width=True):
            model = load_model()
            img_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence, predicted = torch.max(probabilities, 0)
                predicted_class = class_labels[predicted.item()]
                confidence_pct = confidence.item() * 100
            
            # Store prediction in session state
            st.session_state.last_prediction = {
                'disease': predicted_class.replace('___', ' - ').replace('_', ' '),
                'confidence': confidence_pct,
                'is_healthy': "healthy" in predicted_class.lower()
            }
            
            st.subheader("🎯 Prediction Result")
            st.write(f"**Disease:** {st.session_state.last_prediction['disease']}")
            st.write(f"**Confidence:** {confidence_pct:.2f}%")
            
            if st.session_state.last_prediction['is_healthy']:
                st.success("✅ The leaf looks healthy! 🌿")
            else:
                st.error(f"⚠️ Disease detected: {st.session_state.last_prediction['disease']}")
                st.info("💡 Ask the AI Assistant for treatment recommendations!")

# --- Right Column: Chatbot ---
with col2:
    st.header("💬 AI Plant Health Assistant")
    
    if not api_key:
        st.info("👈 Please enter your Gemini API key in the sidebar to use the chatbot.")
    else:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask about plant diseases, treatments, or care tips...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Create context from last prediction if available
            context = ""
            if st.session_state.last_prediction:
                context = f"\n\nContext: The user just uploaded an image and the model detected: {st.session_state.last_prediction['disease']} with {st.session_state.last_prediction['confidence']:.2f}% confidence."
            
            # Prepare prompt for Gemini
            system_prompt = """You are an expert plant pathologist and agricultural assistant. 
            Help users understand plant diseases, provide treatment recommendations, and offer care tips.
            Be concise, practical, and friendly. If asked about a specific disease, provide:
            1. Brief description
            2. Common causes
            3. Treatment options
            4. Prevention tips"""
            
            full_prompt = system_prompt + context + "\n\nUser question: " + user_input
            
            try:
                # Get response from Gemini
                with st.spinner("Thinking..."):
                    response = st.session_state.gemini_model.generate_content(full_prompt)
                    bot_response = response.text
                
                # Add bot response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                
                # Rerun to update chat display
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Made with ❤️ using Streamlit, PyTorch, and Google Gemini</p>
    </div>
    """,
    unsafe_allow_html=True
)