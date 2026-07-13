import gradio as gr
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = TFBertModel.from_pretrained(MODEL_NAME)

def bert_embed(text):
    if not text or not text.strip():
        return {"error": "Please enter some text."}

    inputs = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=128,
    )

    outputs = model(**inputs)
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()[0]

    return {
        "text": text,
        "embedding_shape": list(embeddings.shape),
        "embedding": embeddings[:20].tolist(),
    }

demo = gr.Interface(
    fn=bert_embed,
    inputs=gr.Textbox(lines=3, placeholder="Enter text here..."),
    outputs=gr.JSON(),
    title="BERT Embedding Demo",
    description="A simple Hugging Face Space using BERT with Gradio.",
)

if __name__ == "__main__":
    demo.launch()