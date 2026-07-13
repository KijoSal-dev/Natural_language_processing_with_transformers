import gradio as gr
import torch
from transformers import BertTokenizer, BertModel

MODEL_NAME = "bert-base-uncased"

# Load model
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def bert_embed(text):
    if not text or not text.strip():
        return {"error": "Please enter some text."}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        # Move to CPU for JSON serialization
        if torch.cuda.is_available():
            embedding = embedding.cpu()
        
        embedding = embedding.tolist()

    return {
        "text": text,
        "embedding_shape": [len(embedding)],
        "embedding_first_20_values": embedding[:20],
    }

# Create the interface - MINIMAL version
demo = gr.Interface(
    fn=bert_embed,
    inputs=gr.Textbox(lines=3, label="Enter text"),
    outputs=gr.JSON(label="Output"),
    title="BERT Embedding Demo",
    description="Simple BERT text embedding demo on Hugging Face Spaces.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)