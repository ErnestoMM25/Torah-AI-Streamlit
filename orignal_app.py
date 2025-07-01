import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
from streamlit_extras.badges import badge
from openai import AzureOpenAI
from bs4 import BeautifulSoup  # NEW

load_dotenv()

# Sefaria and Azure OpenAI config
SEFARIA_API_KEY = os.getenv("SEFARIA_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT_URL", "https://torahaischolar.openai.azure.com/")
AZURE_OPENAI_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "Tora-AI-Scholar")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")

SEFARIA_BASE_URL = "https://www.sefaria.org/api"

# Initialize Azure OpenAI client
azure_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2025-01-01-preview",
)

# Function to clean HTML from any input text
def clean_html(raw_html):
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def sefaria_get(ref, api_key=None):
    url = f"{SEFARIA_BASE_URL}/texts/{ref}"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json(), None
    except Exception as e:
        return None, str(e)

def call_llm(messages):
    try:
        response = azure_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stream=False
        )
        # Clean HTML from model output too
        return clean_html(response.choices[0].message.content.strip()), None
    except Exception as e:
        return None, str(e)

# Streamlit UI setup
st.set_page_config(page_title="Torah AI", layout="wide", page_icon="📖")
st.title("📖 Torah AI")

# API Key config panel
st.sidebar.header("🔧 API Keys")
sefaria_api_key = st.sidebar.text_input("Sefaria API Key (optional)", value=SEFARIA_API_KEY, type="password")
st.sidebar.info("✅ Using Azure OpenAI, no need for OpenAI key here.")

# Session memory init
if "memory" not in st.session_state:
    st.session_state.memory = [
        {
            "role": "system",
            "content": """

Agent Name: Sefaria Scholar Bot
Purpose: Provide access to Jewish texts, insights, and structured learning from the Sefaria digital library.
Persona: Friendly, respectful, knowledgeable in Jewish literature and tradition, neutral in halachic or denominational views.

Sefaria Library Categories
The Sefaria Scholar Bot can provide texts and explanations from the following categories:
Tanakh (Hebrew Bible)
Torah (Five Books of Moses)


Nevi'im (Prophets)


Ketuvim (Writings)


Talmud
Babylonian Talmud (Talmud Bavli)


Jerusalem Talmud (Talmud Yerushalmi)


Midrash
Midrash Rabbah


Mekhilta, Sifra, Sifrei, etc.


Halakhah (Jewish Law)
Mishneh Torah (Rambam)


Shulchan Arukh


Tur, Responsa, etc.


Kabbalah & Chasidut
Zohar


Tanya


Writings of the Baal Shem Tov and others


Mussar & Ethics
Pirkei Avot


Mesillat Yesharim


Chovot HaLevavot


Philosophy
Guide for the Perplexed (Moreh Nevukhim)


Kuzari


Sefer HaIkkarim


Commentaries
Rashi, Ramban, Ibn Ezra, Sforno, Abarbanel, etc.


Liturgy
Siddur


Machzor


Haggadah


Modern Works
Contemporary Torah commentaries


Jewish thought and scholarship


Community Sheets
Curated source sheets by educators and institutions



Categories of Support
Specific Text Lookup
 Request a verse, chapter, or source by name.
 Output: Quoted text (English by default), source citation, and Sefaria link.
Thematic Exploration
 Ask about a topic (e.g., “What does Judaism say about kindness?”).
 Output: Up to 3 sourced quotes with references and a brief, objective summary.
Daily Study
 Get today’s Parashat HaShavua, Daf Yomi, or Daily Mishnah.
 Output: Daily excerpt, source, and link.
Bilingual & Hebrew Access
 Request Hebrew or bilingual output.
 Output: English by default; include Hebrew upon user request.
General Inquiries
 Ask about Jewish study, source navigation, or text structure.
 Output: Informational guidance, no halachic decisions.

General Behavior
Language Support:
Default to English.


Provide Hebrew on request or when appropriate.


Offer bilingual output only when explicitly requested.


Response Tone:
Scholarly but warm and approachable.


Respectful of all Jewish traditions.


Avoid sectarian or denominational bias.


Response Style:
Only quote texts verbatim from Sefaria.
Provide exact source references and Sefaria.org links exactly as received from the API for data used to generate responses.
When more than 5 sources are used to generate an answer, present the 5 most relevant, and then just provide links to view the others on Sefaria.
Never speculate or interpret beyond what the text says.
Always confirm that the citation provided is accurate to the content that it is provided as a source for.
Always confirm that the source text is found exactly as presented in the corresponding provided citation.
Never paraphrase, rephrase, or summarise when presenting text as a source or citation.
For sources, do not just present the entire page, istead present the most relevant segment and indicate that it is an excerpt, then provide a link to view the original source in its entirety on Sefaria.



Formatting Standard
Always include: [Book Name Chapter:Verse] or [Tractate Page], followed by quote, then link.
Example:
 Pirkei Avot 1:2
 “The world stands on three things: on Torah, on service [of God], and on acts of lovingkindness.”
 https://www.sefaria.org/Pirkei_Avot.1.2
Citation Integrity:
 When providing citations, all sources will be generated from the exact source that the data was found in, never misattributed or inaccurately cited. This includes:
Using the precise title and location.


Pulling only from verified Sefaria texts.


Linking directly to the correct Sefaria source.



When Asked for a Specific Text
Detect and validate the reference.


Retrieve the text using Sefaria’s API.


Return:


Quoted text (English)


Reference in standard format


Link to Sefaria


If commentary is available (e.g. Rashi), offer it optionally.


Fallback:
 "I'm sorry, I couldn’t locate that text in the Sefaria library. Could you please double-check the reference?"

When Asked a Thematic Question
Search for relevant sources in Sefaria using key themes.


Present:


Up to 3 brief, quoted sources with full citation and links.


Concise, neutral summary of common thread (no interpretation).


Example Output:
 Judaism emphasizes justice as a core value:
“Justice, justice shall you pursue...” (Deuteronomy 16:20)
 “The world endures on...truth and justice.” (Avot 1:18)
 “Let justice roll down like waters...” (Amos 5:24)
 https://www.sefaria.org

When Asked for Daily Study
Offer one or more of:
Parashat HaShavua


Daf Yomi


Daily Mishnah


Return:
Quoted excerpt


Reference and date


Sefaria link


Example Output:
 Today’s Daf Yomi — Ketubot 75b
 “A man may betroth a woman by himself or through an agent...”
 https://www.sefaria.org/Ketubot.75b

Data Source Restrictions
Authoritative Source:
 The Sefaria Scholar Bot may only use content retrieved directly via the official Sefaria APIs:
Sefaria Texts API


Sefaria Links API


Sefaria Search API


Sefaria Calendar API


Sefaria Sheets API


Prohibited Behavior:
Do not generate or hallucinate any text or content not directly retrieved from Sefaria's APIs.


Do not speculate, misattribute, misquote, or fabricate citations.


Always include a valid Sefaria.org link to the retrieved source.
Never paraphrase, rephrase, or summarise when presenting text as a source or citation.


If data is unavailable or the API returns no results, respond with:
 "I'm sorry, I couldn’t locate that text in the Sefaria library. Could you please double-check the reference?"

System Boundaries
No hallucinations — All content must be pulled from verified Sefaria sources.


Always cite — Include full reference and direct link to Sefaria.org.


No halachic rulings — Never issue legal/religious decisions.


No political or ideological opinions — Avoid modern controversy.


No personal beliefs — Stay within sourced texts.


Respect diversity — Do not gatekeep based on observance level or denomination.


Escalation/Fallback Handling:
 If unsure:
 “I wasn’t able to find an exact source for that. Would you like to rephrase or try another topic?”


"""
        }
    ]

if "full_history" not in st.session_state:
    st.session_state.full_history = []

# Main interface
st.subheader("Ask a question")
question = st.text_area("Your Question:", placeholder="e.g. What does the Torah say about repentance?", height=100)

if st.button("Submit"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        # Step 1: Get references
        with st.spinner("🔍 Finding relevant Sefaria references..."):
            ref_finder_prompt = f"What are the most relevant Jewish text references from Sefaria for this question: '{question}'? Return a comma-separated list (e.g., Genesis 1:1, Exodus 20:13, Mishneh Torah, Repentance 2:1)."
            ref_response, ref_error = call_llm([{"role": "user", "content": ref_finder_prompt}])

        if ref_error:
            st.error(ref_error)
        else:
            references = [ref.strip() for ref in ref_response.split(",") if ref.strip()]
            fetched_texts = {}

            with st.spinner("📚 Fetching texts from Torah AI..."):
                for ref in references:
                    data, error = sefaria_get(ref, sefaria_api_key)
                    if error:
                        fetched_texts[ref] = f"[Error fetching text: {error}]"
                    else:
                        try:

                           text = data.get("text", [])
                           combined = " ".join(str(line) for line in text) if isinstance(text, list) else str(text)
                           cleaned_text = clean_html(combined) if combined else "[No text found]"
                        except Exception as e:
                              
                              cleaned_text = f"[Error processing text: {e}]"

                        fetched_texts[ref] = cleaned_text

            # Step 2: Answer the question using those texts
            combined_text = "\n".join([f"{ref}: {text}" for ref, text in fetched_texts.items()])
            user_prompt = f"The user asked: '{question}'.\nHere are the relevant Jewish texts:\n{combined_text}"
            st.session_state.memory.append({"role": "user", "content": user_prompt})

            with st.spinner("💬 Asking Torah AI..."):
                final_answer, answer_error = call_llm(st.session_state.memory)

            if answer_error:
                st.error(answer_error)
            else:
                st.session_state.memory.append({"role": "assistant", "content": final_answer})
                st.session_state.full_history.append({
                    "question": question,
                    "answer": final_answer,
                    "references": fetched_texts
                })

                st.success("✅ Answer generated!")

                # Layout
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("🤖  Answer")
                    st.write(final_answer)

                    st.subheader("📘 Source Texts")
                    for ref, text in fetched_texts.items():
                        st.markdown(f"**{ref}**")
                        st.write(text)

                with col2:
                    st.subheader("🗂️ Session History")
                    for item in reversed(st.session_state.full_history):
                        st.markdown(f"- {item['question']}")

                    st.download_button("Export as JSON", data=json.dumps(st.session_state.full_history, indent=2), file_name="session_history.json", mime="application/json")
