import gradio as gr
import torch
from transformers import BertTokenizer, BertModel

MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
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

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    return {
        "text": text,
        "embedding_shape": [len(embedding)],
        "embedding_first_20_values": embedding[:20],
    }

demo = gr.Interface(
    fn=bert_embed,
    inputs=gr.Textbox(lines=3, label="Enter text"),
    outputs=gr.JSON(label="Output"),
    title="BERT Embedding Demo",
    description="Simple BERT text embedding demo on Hugging Face Spaces.",
)

if __name__ == "__main__":
    demo.launch()