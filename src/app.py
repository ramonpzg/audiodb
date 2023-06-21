import streamlit as st
from qdrant_client import QdrantClient
from transformers import pipeline
from audiocraft.models import MusicGen
from panns_inference import AudioTagging
import os

st.title("Music Recommendation App")
st.subheader("A :red[Generative AI]-to-Real Music Approach")

st.markdown("""
The purpose of this app is to help creative people explore the possibilities of Generative AI in the music
domain, while comparing their creations to music made by people with all sorts of instruments.  

There are several moving parts to this app and the most important ones are `transformers`, `audiocraft`, and 
Qdrant for our vector database.
""")

client     = QdrantClient(
    "https://394294d5-30bb-4958-ad1a-15a3561edce5.us-east-1-0.aws.cloud.qdrant.io:6333", 
    api_key=os.environ['QDRANT_API_KEY'],
)
classifier = pipeline("audio-classification", model="ramonpzg/wav2musicgenre")#.to(device)
model      = MusicGen.get_pretrained('small')

# param1, param2 = st.columns(2)
val1 = st.slider("How many seconds?", 5.0, 30.0, value=5.0, step=0.5)
# val2 = param2.slider(
#     "How many inference steps?", 5, 100, value=5, 
#     help="The higher the number, the better the quality of the sound but the longer it takes for your music to be generated."
# )

model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=val1
)

music_prompt = st.text_input(
    label="Music Prompt",
    value="Fast-paced bachata in the style of Romeo Santos."
)

if st.button("Generate Some Music!"):
    with st.spinner("Wait for it..."):
        output = model.generate(descriptions=[music_prompt],progress=True)[0, 0, :].cpu().numpy()
        st.success("Done! :)")

    st.audio(output, sample_rate=32000)

    genres = classifier(output)

    if genres:
        st.markdown("## Best Prediction")
        col1, col2 = st.columns(2, gap="small")
        col1.subheader(genres[0]['label'])
        col2.metric(label="Score", value=f"{genres[0]['score']*100:.2f}%")

        st.markdown("### Other Predictions")
        col3, col4 = st.columns(2, gap="small")
        for idx, genre in enumerate(genres[1:]):
            if idx % 2 == 0:
                col3.metric(label=genre['label'], value=f"{genre['score']*100:.2f}%")
            else:
                col4.metric(label=genre['label'], value=f"{genre['score']*100:.2f}%")

    at = AudioTagging(checkpoint_path=None)
    clipwise_output, embedding = at.inference(output[None, :])

    # features = classifier.feature_extractor(output)

    # with torch.no_grad():
    #     vectr = classifier.model(**features, output_hidden_states=True).hidden_states[-1].mean(dim=1)[0]

    vectr = embedding[0]

    results = client.search(
        collection_name="music_vectors",
        query_vector=vectr,
        limit=10
    )

    st.markdown("## Real Recommendations")

    col5, col6 = st.columns(2)

    for idx, result in enumerate(results):
        if idx % 2 == 0:
            col5.header(f"Genre: {result.payload['genre']}")
            col5.markdown(f"### Artist: {result.payload['artist']}")
            col5.markdown(f"#### Song name: {result.payload['name']}")
            col5.audio(result.payload["urls"])
        else:
            col6.header(f"Genre: {result.payload['genre']}")
            col6.markdown(f"### Artist: {result.payload['artist']}")
            col6.markdown(f"#### Song name: {result.payload['name']}")
            col6.audio(result.payload["urls"])