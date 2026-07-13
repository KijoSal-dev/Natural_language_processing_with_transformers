import gradio as gr
import torch
import warnings
from transformers import BertTokenizer, BertModel
import spaces

# Suppress warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "bert-base-uncased"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Load model and move to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 Using device: {device}")

model = BertModel.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

# The @spaces.GPU decorator ensures proper GPU handling in Spaces
@spaces.GPU
def bert_embed(text):
    """
    Generate BERT embeddings for input text
    """
    if not text or not text.strip():
        return {"error": "Please enter some text."}
    
    try:
        # Tokenize the input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling to get a single embedding vector
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            # Move to CPU for JSON serialization
            embedding = embedding.cpu().tolist()
        
        return {
            "text": text,
            "embedding_shape": [len(embedding)],
            "embedding_first_20_values": embedding[:20],
            "device_used": str(device),
            "embedding_dimension": len(embedding)
        }
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Create a more informative interface
demo = gr.Interface(
    fn=bert_embed,
    inputs=gr.Textbox(
        lines=5, 
        label="Enter text", 
        placeholder="Type or paste your text here...",
        info="Enter any text to get BERT embeddings"
    ),
    outputs=gr.JSON(
        label="Embedding Output",
        info="BERT embeddings and metadata"
    ),
    title="BERT Embedding Demo",
    description="Simple BERT text embedding demo on Hugging Face Spaces.",
    examples=[
        ["Hello, how are you today?"],
        ["The quick brown fox jumps over the lazy dog."],
        ["Artificial intelligence is transforming the world."]
    ],
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )