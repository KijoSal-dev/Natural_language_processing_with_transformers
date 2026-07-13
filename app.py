import gradio as gr
import torch
import warnings
from transformers import BertTokenizer, BertModel

# Suppress warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "bert-base-uncased"

# Check if CUDA is available
if torch.cuda.is_available():
    print("✅ GPU is available, using CUDA")
    device = torch.device("cuda")
else:
    print("ℹ️ GPU not available, using CPU")
    device = torch.device("cpu")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)

# Move model to the correct device
model = model.to(device)
model.eval()

print(f"📱 Model loaded on: {device}")

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
            # Use CLS token embedding (alternative to mean pooling)
            embedding = outputs.pooler_output.squeeze()
            
            # Move to CPU for JSON serialization
            if torch.cuda.is_available():
                embedding = embedding.cpu()
            
            embedding = embedding.tolist()
        
        return {
            "text": text,
            "text_length": len(text),
            "embedding_dimension": len(embedding),
            "embedding_first_10_values": embedding[:10],
            "embedding_summary": f"Vector of {len(embedding)} dimensions",
            "device_used": str(device)
        }
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Create the interface - REMOVED 'info' parameter from gr.JSON
demo = gr.Interface(
    fn=bert_embed,
    inputs=gr.Textbox(
        lines=3, 
        label="Enter text", 
        placeholder="Type your text here...",
        info="Enter any text to get BERT embeddings"
    ),
    outputs=gr.JSON(label="Output"),  # Removed 'info' parameter
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