import streamlit as st
from diffusers import AudioLDMPipeline
import torch
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models
import pandas as pd
from transformers import AutoModel, pipeline, AutoFeatureExtractor



st.title("Music Recommendation App")
st.subheader("A :red[Generative AI]-to-Real Music Approach")

st.markdown("""
The purpose of this app is to help creative people explore the possibilities of Generative AI in the music
domain, while comparing their creations to music made by people with all sorts of instruments.  

There are several moving parts to this app and the most important ones are `transformers`, `diffusers`, and 
Qdrant for our vector database.
""")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
repo_id = "cvssp/audioldm"
pipe = AudioLDMPipeline.from_pretrained(repo_id)
# pipe = pipe.to("cuda")


param1, param2 = st.columns(2)
val1 = param1.slider("How many seconds?", 5.0, 120.0, value=5.0, step=0.5)
val2 = param2.slider(
    "How many inference steps?", 5, 100, value=5, 
    help="The higher the number, the better the quality of the sound but the longer it takes for your music to be generated."
)

music_prompt = st.text_input(
    label="Music Prompt",
    value="Techno music with a strong, upbeat tempo and high melodic riffs."
)


audio = pipe(
    prompt=music_prompt, num_inference_steps=val2, audio_length_in_s=val1
).audios[0]

st.audio(audio, sample_rate=16000)

classifier = pipeline("audio-classification", model="notebooks/sec_mod")

genre = classifier(audio)

st.markdown("## Best Prediction")
m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(label=genre[0]['label'], value=f"{genre[0]['score']*100:.2f}%")
m2.metric(label=genre[1]['label'], value=f"{genre[1]['score']*100:.2f}%")
m3.metric(label=genre[2]['label'], value=f"{genre[2]['score']*100:.2f}%")
m4.metric(label=genre[3]['label'], value=f"{genre[3]['score']*100:.2f}%")
m5.metric(label=genre[4]['label'], value=f"{genre[4]['score']*100:.2f}%")


model = AutoModel.from_pretrained('notebooks/sec_mod').to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained("notebooks/sec_mod")
inputs = feature_extractor(
    audio, sampling_rate=feature_extractor.sampling_rate, 
    return_tensors="pt", max_length=16000, truncation=True
)

with torch.no_grad():
    last_hidden_state = model(**inputs.to(device)).last_hidden_state[:, 0]
last_hidden_state.size()


st.markdown("## Real Recommendations")

client = QdrantClient("localhost", port=6333)
vectr = last_hidden_state.cpu().numpy()[0, :]

results = client.search(
    collection_name="test_collection",
    query_vector=vectr,
    limit=10
)

col1, col2 = st.columns(2)

with col1:
    st.header(f"Genre: {results[0].payload['genre']}")
    st.subheader(f"Artist: {results[0].payload['artist']}")
    st.audio(results[0].payload["audio_path"])
    
    st.header(f"Genre: {results[1].payload['genre']}")
    st.subheader(f"Artist: {results[1].payload['artist']}")
    st.audio(results[1].payload["audio_path"])
    
    st.header(f"Genre: {results[2].payload['genre']}")
    st.subheader(f"Artist: {results[2].payload['artist']}")
    st.audio(results[2].payload["audio_path"])
    
    st.header(f"Genre: {results[3].payload['genre']}")
    st.subheader(f"Artist: {results[3].payload['artist']}")
    st.audio(results[3].payload["audio_path"])
    
    st.header(f"Genre: {results[4].payload['genre']}")
    st.subheader(f"Artist: {results[4].payload['artist']}")
    st.audio(results[4].payload["audio_path"])

with col2:
    st.header(f"Genre: {results[5].payload['genre']}")
    st.subheader(f"Artist: {results[5].payload['artist']}")
    st.audio(results[5].payload["audio_path"])
    
    st.header(f"Genre: {results[6].payload['genre']}")
    st.subheader(f"Artist: {results[6].payload['artist']}")
    st.audio(results[6].payload["audio_path"])
    
    st.header(f"Genre: {results[7].payload['genre']}")
    st.subheader(f"Artist: {results[7].payload['artist']}")
    st.audio(results[7].payload["audio_path"])
    
    st.header(f"Genre: {results[8].payload['genre']}")
    st.subheader(f"Artist: {results[8].payload['artist']}")
    st.audio(results[8].payload["audio_path"])
    
    st.header(f"Genre: {results[9].payload['genre']}")
    st.subheader(f"Artist: {results[9].payload['artist']}")
    st.audio(results[9].payload["audio_path"])