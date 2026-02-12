import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import pickle
from torchvision import models, transforms
import numpy as np

# Set page config
st.set_page_config(page_title="Neural Storyteller", page_icon="🖼️", layout="wide")

# Define model classes
class Encoder(nn.Module):
    def __init__(self, image_feature_size=2048, hidden_size=512):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(image_feature_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, image_features):
        hidden = self.fc(image_features)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=1):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, captions, hidden):
        embeddings = self.embed(captions)
        embeddings = self.dropout(embeddings)
        hidden = hidden.unsqueeze(0)
        cell = torch.zeros_like(hidden)
        lstm_out, _ = self.lstm(embeddings, (hidden, cell))
        outputs = self.fc(lstm_out)
        return outputs

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = Encoder(2048, hidden_size)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size)
    
    def forward(self, image_features, captions):
        hidden = self.encoder(image_features)
        outputs = self.decoder(captions, hidden)
        return outputs

# Load model and vocabulary
@st.cache_resource
def load_model_and_vocab():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load vocabulary
    with open('vocab.pkl', 'rb') as f:
        vocab_data = pickle.load(f)
        word2idx = vocab_data['word2idx']
        idx2word = vocab_data['idx2word']
        vocab_size = vocab_data['vocab_size']
    
    # Load model
    model = ImageCaptioningModel(vocab_size, 256, 512)
    model.load_state_dict(torch.load('image_captioning_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # Load ResNet50 for feature extraction
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.to(device)
    resnet.eval()
    
    return model, word2idx, idx2word, resnet, device

# Extract features from image
def extract_features(image, resnet, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = resnet(img_tensor).squeeze()
    
    return features

# Generate caption using greedy search
def generate_caption(model, image_feature, word2idx, idx2word, max_len=50):
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        image_feature = image_feature.unsqueeze(0).to(device)
        hidden = model.encoder(image_feature)
        
        current_word_idx = word2idx['<start>']
        caption = []
        input_seq = [current_word_idx]
        
        for _ in range(max_len):
            input_tensor = torch.LongTensor([input_seq]).to(device)
            outputs = model.decoder(input_tensor, hidden)
            predicted_idx = outputs[0, -1, :].argmax().item()
            
            if predicted_idx == word2idx['<end>']:
                break
            
            if predicted_idx not in [word2idx['<pad>'], word2idx['<start>'], word2idx['<unk>']]:
                caption.append(idx2word[predicted_idx])
            
            input_seq.append(predicted_idx)
        
        return ' '.join(caption)

# Main app
def main():
    st.title("Neural Storyteller - Image Captioning")
    st.markdown("### Upload an image and let AI describe it!")
    
    # Load model
    with st.spinner("Loading model..."):
        model, word2idx, idx2word, resnet, device = load_model_and_vocab()
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses a Seq2Seq model with:\n"
        "- **Encoder**: ResNet50 + Linear layer\n"
        "- **Decoder**: LSTM network\n"
        "- **Vocabulary**: 7,727 words\n"
        "- **Parameters**: 8.5M"
    )
    
    st.sidebar.header("Model Performance")
    st.sidebar.metric("BLEU-4 Score", "0.0423")
    st.sidebar.metric("Precision", "0.3021")
    st.sidebar.metric("Recall", "0.2538")
    st.sidebar.metric("F1-Score", "0.2635")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        with col2:
            with st.spinner('Generating caption...'):
                # Extract features
                features = extract_features(image, resnet, device)
                
                # Generate caption
                caption = generate_caption(model, features, word2idx, idx2word)
            
            st.success("Caption Generated!")
            st.markdown(f"### **{caption.capitalize()}**")
            
            # Additional info
            with st.expander("More Details"):
                st.write(f"**Caption Length**: {len(caption.split())} words")
                st.write(f"**Device**: {device}")
    
    else:
        st.info(" Please upload an image to get started!")
        
        # Show example
        st.markdown("### Example Output:")
        st.image("caption_examples.png", use_container_width=True)

if __name__ == "__main__":
    main()
