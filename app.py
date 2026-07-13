import spaces  # MUST be the first import — before torch/transformers, required by ZeroGPU

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


def _embed(text):
    """Internal helper: returns the mean-pooled BERT embedding as a torch tensor on `device`."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

    return embedding


@spaces.GPU
def bert_embed(text):
    if not text or not text.strip():
        return {"error": "Please enter some text."}

    embedding = _embed(text)

    if torch.cuda.is_available():
        embedding = embedding.cpu()
    embedding = embedding.tolist()

    return {
        "text": text,
        "embedding_shape": [len(embedding)],
        "embedding_first_20_values": embedding[:20],
    }


@spaces.GPU
def bert_similarity(text1, text2):
    if not text1 or not text1.strip() or not text2 or not text2.strip():
        return {"error": "Please enter text in both boxes."}

    emb1 = _embed(text1)
    emb2 = _embed(text2)

    # Cosine similarity between the two mean-pooled embeddings
    similarity = torch.nn.functional.cosine_similarity(
        emb1.unsqueeze(0), emb2.unsqueeze(0)
    ).item()

    return {
        "text_1": text1,
        "text_2": text2,
        "cosine_similarity": round(similarity, 4),
    }


# Tab 1: single text embedding
embed_interface = gr.Interface(
    fn=bert_embed,
    inputs=gr.Textbox(lines=3, label="Enter text"),
    outputs=gr.JSON(label="Output"),
    title="BERT Embedding",
    description="Get the BERT embedding vector for a single piece of text.",
)

# Tab 2: cosine similarity between two texts
similarity_interface = gr.Interface(
    fn=bert_similarity,
    inputs=[
        gr.Textbox(lines=3, label="Text 1"),
        gr.Textbox(lines=3, label="Text 2"),
    ],
    outputs=gr.JSON(label="Output"),
    title="BERT Cosine Similarity",
    description="Compare two texts using cosine similarity of their BERT embeddings.",
)

demo = gr.TabbedInterface(
    [embed_interface, similarity_interface],
    ["Embedding", "Similarity"],
    title="BERT Embedding Demo",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)