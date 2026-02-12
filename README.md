# neural-storyteller-image-captioning
Image captioning using Seq2Seq with ResNet50 and LSTM - Generative AI Assignment


# 🖼️ Neural Storyteller - Image Captioning with Seq2Seq

AI-powered image captioning system using deep learning to generate natural language descriptions for images.

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| BLEU-1 | 0.2689 |
| BLEU-2 | 0.1421 |
| BLEU-3 | 0.0830 |
| BLEU-4 | 0.0477 |
| Precision | 0.3107 |
| Recall | 0.2642 |
| F1-Score | 0.2739 |

## 🏗️ Architecture

- **Encoder:** ResNet50 → Linear (2048 → 512)
- **Decoder:** LSTM (512 hidden units)
- **Vocabulary:** ~7,700 words
- **Dataset:** Flickr30k (31,783 images)

## 🚀 Training Details

- **Epochs:** 30
- **Batch Size:** 32
- **Learning Rate:** 0.0001
- **Optimizer:** Adam with ReduceLROnPlateau
- **GPU:** Kaggle T4 x2

## 💻 Technologies

- PyTorch
- torchvision
- NLTK
- Streamlit
- NumPy, Pandas

## 📝 Assignment

**Course:** Generative AI (AI4009)  
**Institution:** NUCES  
**Semester:** Spring 2026
