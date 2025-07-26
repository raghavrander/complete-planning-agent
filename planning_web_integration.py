import streamlit as st
import os
import requests
import json
from groq import Groq
import datetime

# --- ML and NLP Imports ---
import torch
from sentence_transformers import SentenceTransformer, util

# --- Web Scraping and Search Imports ---
from googlesearch import search
from bs4 import BeautifulSoup


# --- Initialization and Configuration ---
try:
    # Attempt to get secrets from Streamlit's secrets manager
    FOURSQUARE_API_KEY = st.secrets["FOURSQUARE_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError):
    # Fallback for local development using a .env file
    from dotenv import load_dotenv
    load_dotenv()
    FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

FOURSQUARE_API_URL = "https://places-api.foursquare.com/places/"
MAJOR_CHAINS = ["starbucks", "mcdonald's", "subway", "costa coffee", "dunkin'"]

st.set_page_config(page_title="Agentic Itinerary Builder", page_icon="🗺️", layout="centered")

# --- Caching the Machine Learning Model ---
@st.cache_resource
def load_similarity_model():
    """Loads and caches the sentence-transformer model."""
    print("INFO: Loading sentence-transformer model for the first time...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("INFO: Model loaded successfully.")
    return model

# --- CSS and Styling ---
st.markdown("""
    <style>
        div[data-testid="stTextInput"] input {
            border: 2px solid #777; border-radius: 10px; padding: 10px;
            background-color: #f0f2f6; color: #333;
        }
        div[data-testid="stTextInput"] input::placeholder { color: #888; opacity: 1; }
        .stChatMessage { border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
        .stChatMessage.user { background-color: #e1f5fe; }
        .stChatMessage.assistant { background-color: #f1f8e9; }
    </style>
    """, unsafe_allow_html=True)


# --- Tool Definition (Find Places) ---
def find_places(query: str, radius: int = 2500) -> dict:
    """Finds hidden gem places using Foursquare API with semantic filtering."""
    model = load_similarity_model()
    SIMILARITY_THRESHOLD = 0.4
    location = st.session_state.get('location')
    if not location:
        return {"error": "Fatal Error: Location not found in session state."}
    lat, lon = location['coords']['latitude'], location['coords']['longitude']

    params = {"ll": f"{lat},{lon}", "query": query, "radius": radius, "limit": 10, "fields": "name,categories,location,distance,chains"}
    headers = {"X-Places-Api-Version": "2025-06-17", "accept": "application/json", "Authorization": FOURSQUARE_API_KEY}

    try:
        response = requests.get(f"{FOURSQUARE_API_URL}search", params=params, headers=headers, timeout=15)
        response.raise_for_status()
        api_results = response.json().get('results', [])
        semantically_filtered_gems = []
        query_embedding = model.encode(query, convert_to_tensor=True)

        for place in api_results:
            if any(chain in place.get("name", "").lower() for chain in MAJOR_CHAINS) or place.get("chains"):
                continue
            if place.get('categories'):
                primary_category = place['categories'][0]
                category_name = primary_category.get('name', '')
                if not category_name: continue

                category_embedding = model.encode(category_name, convert_to_tensor=True)
                cosine_score = util.pytorch_cos_sim(query_embedding, category_embedding).item()

                if cosine_score > SIMILARITY_THRESHOLD:
                    icon_data = primary_category.get('icon')
                    photo_url = f"{icon_data['prefix']}bg_88{icon_data['suffix']}" if icon_data else "https://placehold.co/128x128/27272a/FFFFFF?text=Icon"

                    semantically_filtered_gems.append({
                        "name": place.get("name", "Name not available"),
                        "category": primary_category.get('name', 'Place'),
                        "distance": f"{(place.get('distance', 0) / 1000):.1f}km away",
                        "address": place.get("location", {}).get("formatted_address", "N/A"),
                        "photo_url": photo_url,
                    })

        if not semantically_filtered_gems:
            return {"error": f"I couldn't find any relevant places for '{query}'."}
        return {"results": semantically_filtered_gems}
    except Exception as e:
        return {"error": f"An API error occurred for query '{query}': {e}"}

# ===================================================================
# --- AGENTIC CORE ---
# ===================================================================
client = Groq(api_key=GROQ_API_KEY)

# --- Web Search Tool (Corrected and Robust) ---
def web_search(query: str) -> dict:
    """Performs a web search, finds the first valid external link, and scrapes its content."""
    st.write(f"🔎 Researching: '{query}'...")
    try:
        search_results = search(query, num_results=5, lang="en")
        top_result_url = None
        for url in search_results:
            if url.startswith("http"):
                top_result_url = url
                break

        if not top_result_url:
            return {"error": "Could not find a relevant web page after checking top results."}

        st.write(f"Found page: {top_result_url}")
        st.write("Reading content...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(top_result_url, timeout=15, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        paragraphs = soup.find_all('p')
        page_text = ' '.join(p.get_text() for p in paragraphs)
        return {"url": top_result_url, "text": page_text[:4000]}
    except Exception as e:
        return {"error": f"Failed to search or scrape the web: {e}"}

# --- Research Agent Brain ---
def run_research_agent(user_question: str, context: dict) -> str:
    """Answers a question based on provided web content."""
    st.write("🤔 Analyzing the information...")
    research_prompt = f"""You are a helpful research assistant. Your task is to answer the user's question based *only* on the provided text from a webpage.
    Be concise and directly answer the question. If the text doesn't contain the answer, say that you couldn't find the information in the provided context.

    CONTEXT FROM {context.get('url', 'the web')}:
    ---
    {context.get('text', 'No text found.')}
    ---

    USER'S QUESTION: "{user_question}"

    YOUR ANSWER:
    """
    try:
        response = client.chat.completions.create(messages=[{"role": "user", "content": research_prompt}], model="llama3-70b-8192")
        return response.choices[0].message.content
    except Exception as e:
        return f"I had trouble analyzing the information. Error: {e}"

# --- Finder Agent ---
def run_finder_agent(user_prompt: str) -> dict:
    """Extracts a keyword and calls the find_places tool."""
    system_prompt = """You are a keyword extractor AI. Your only job is to extract the single most important noun or activity from the user's request. You must ignore all other words.
    **Critical Rules:**
    1. Your output MUST be a single, valid JSON object and nothing else.
    2. The JSON format is: `{"tool": "find_places", "parameters": {"query": "KEYWORD"}}`
    3. **You MUST remove descriptive adjectives.**
    **Examples:**
    User: "I want to go to a quiet park" -> {"tool": "find_places", "parameters": {"query": "park"}}
    User: "Find me a nice bookstore" -> {"tool": "find_places", "parameters": {"query": "bookstore"}}
    """
    try:
        chat_completion = client.chat.completions.create(messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], model="llama3-8b-8192", temperature=0.0, response_format={"type": "json_object"})
        action = json.loads(chat_completion.choices[0].message.content)
        if isinstance(action, dict) and action.get("tool") == "find_places":
            return find_places(**action["parameters"])
        return {"error": "Finder agent could not determine a place to search for."}
    except Exception as e:
        return {"error": f"Finder agent failed: {e}"}

# --- Planning Agent ---
def run_planning_agent(user_prompt: str) -> dict:
    """Creates a multi-step plan based on a user request."""
    st.write("🤔 Thinking about a plan for you...")
    plan_prompt = f"""You are an expert travel planner. Create a logical, step-by-step plan based on the user's request. The output MUST be a JSON object containing a list of steps. Each step must be a simple query to find a place.
    **User Request:** "{user_prompt}"
    **Example for "Plan my day":** {{"plan": [{{"step_description": "First, let's get some breakfast.", "tool_query": "breakfast restaurant"}}, {{"step_description": "Next, a fun morning activity.", "tool_query": "museum or park"}}, {{"step_description": "Time for lunch!", "tool_query": "lunch restaurant"}}]}}
    """
    try:
        plan_completion = client.chat.completions.create(messages=[{"role": "system", "content": "You are an expert travel planner that only outputs JSON."}, {"role": "user", "content": plan_prompt}], model="llama3-70b-8192", temperature=0.1, response_format={"type": "json_object"})
        plan_steps = json.loads(plan_completion.choices[0].message.content).get("plan", [])
    except Exception as e:
        return {"error": f"Sorry, I had trouble creating a plan. Error: {e}"}
    itinerary_items = []
    for step in plan_steps:
        st.write(f"🔍 {step['step_description']}")
        result = find_places(query=step['tool_query'])
        if "results" in result and result["results"]:
            itinerary_items.append({"step": step['step_description'], "place": result["results"][0]})
        else:
            itinerary_items.append({"step": step['step_description'], "place": {"name": f"Could not find a place for: {step['tool_query']}", "category": "N/A"}})
    return {"itinerary": itinerary_items}

# --- Itinerary Synthesizer ---
def synthesize_itinerary(itinerary_items: list) -> str:
    """Turns a structured list of places into a friendly narrative."""
    synthesis_prompt = f"""You are a friendly travel guide. Turn this structured list into a fun, engaging, and easy-to-read travel plan. Use Markdown for formatting.
    Here are the places I found for the user: {json.dumps(itinerary_items, indent=2)}
    Now, create the final itinerary narrative for the user."""
    try:
        synthesis_completion = client.chat.completions.create(messages=[{"role": "user", "content": synthesis_prompt}], model="llama3-70b-8192", temperature=0.5)
        return synthesis_completion.choices[0].message.content
    except Exception as e:
        return f"Could not generate a summary. Error: {e}"

# --- NEW: Conversational Agent ---
def run_conversational_agent(user_prompt: str) -> str:
    """Handles general conversation and guides the user towards the agent's capabilities."""
    st.write("💬 Formulating a helpful response...")
    conversational_prompt = f"""You are a friendly and helpful travel and discovery assistant.
    Your goal is to understand the user's vague prompt and guide them towards one of your core functions: planning an itinerary, finding a place, or researching a topic.
    Be concise, friendly, and proactive.

    Examples:
    User prompt: "hey help me"
    Your Answer: "Of course! I'd be happy to help. Are you looking to plan a multi-stop itinerary, find a specific type of place, or research some interesting facts about a location?"

    User prompt: "i am bored"
    Your Answer: "I can definitely help with that! A new experience is a great cure for boredom. What kind of place would you be interested in visiting? For example, we could find a park, a museum, or a great coffee shop."

    User prompt: "hi"
    Your Answer: "Hello! How can I help you plan your day or find something interesting nearby?"

    Now, please respond to the following user prompt.
    User prompt: "{user_prompt}"
    Your Answer:
    """
    try:
        response = client.chat.completions.create(messages=[{"role": "user", "content": conversational_prompt}], model="llama3-70b-8192", temperature=0.6)
        return response.choices[0].message.content
    except Exception as e:
        return f"I'm sorry, I had a little trouble thinking of a response. Error: {e}"

# --- UPDATED: Router Agent with "conversational" capability ---
def run_router_agent(user_prompt: str) -> str:
    """Classifies user intent as 'planner', 'finder', 'research', or 'conversational'."""
    router_prompt = f"""You are an intent classification agent. Your only job is to determine the user's intent.
    Respond with ONLY one word: "planner", "finder", "research", or "conversational".

    - "planner": The request is high-level or involves a sequence (e.g., "plan my day", "fun afternoon").
    - "finder": The request asks to find a specific category of place (e.g., "park", "tacos", "bookstore").
    - "research": The request is a specific question asking for facts (e.g., "what is the Eiffel Tower", "history of the Colosseum").
    - "conversational": The request is a greeting, a vague statement of need, or a general question about the bot's capabilities (e.g., "hi", "help me", "i am bored", "what can you do?").

    User request: "{user_prompt}" """
    try:
        router_completion = client.chat.completions.create(messages=[{"role": "system", "content": "You are an intent classification agent that only responds with a single word: 'planner', 'finder', 'research', or 'conversational'."}, {"role": "user", "content": router_prompt}], model="llama3-8b-8192", temperature=0.0)
        return router_completion.choices[0].message.content.strip().lower()
    except Exception:
        return "conversational" # Default to conversational on error

# --- UI Components ---
def suggestion_card(result: dict):
    with st.container(border=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(result["photo_url"], width=88)
        with col2:
            st.subheader(result["name"])
            st.caption(f"{result['category']} · {result['distance']}")
        st.write("")
        st.caption(result["address"])

# --- Main Application UI ---
st.title("🤖 Agentic Itinerary Builder")
load_similarity_model()

if not FOURSQUARE_API_KEY or not GROQ_API_KEY:
    st.error("🔴 API Key Missing! Please add FOURSQUARE_API_KEY and GROQ_API_KEY to your Streamlit secrets.", icon="🚨")
else:
    if 'location' not in st.session_state:
        st.session_state.location = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if st.session_state.location is None:
        from streamlit_js_eval import get_geolocation
        st.session_state.location = get_geolocation()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type") == "cards":
                st.subheader("Here's what I found for you:")
                col1, col2 = st.columns(2)
                for i, result in enumerate(message["content"]):
                    with (col1 if i % 2 == 0 else col2):
                        suggestion_card(result)
            elif message.get("type") in ["itinerary", "research"]:
                st.markdown(message["content"])
            elif message.get("type") == "error":
                st.error(message["content"])
            else: # Simple user or assistant text message
                st.markdown(message["content"])

    # Chat input and main agent logic
    if prompt := st.chat_input("Plan my day, research a topic, or find a place..."):
        if st.session_state.location:
            st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Let me think..."):
                    agent_type = run_router_agent(prompt)
                    st.write(f"Routing to: **{agent_type}**...")

                    # --- Sanity Check to prevent incorrect fall-through ---
                    VALID_AGENTS = ["planner", "finder", "research", "conversational"]
                    if agent_type not in VALID_AGENTS:
                        st.write(f"Router returned an unexpected value ('{agent_type}'). Defaulting to conversational agent.")
                        agent_type = "conversational"

                    # Execute the chosen agent
                    if agent_type == "conversational":
                        response = run_conversational_agent(prompt)
                        st.session_state.messages.append({"role": "assistant", "content": response, "type": "text"})

                    elif agent_type == "planner":
                        planner_result = run_planning_agent(prompt)
                        if "itinerary" in planner_result:
                            final_itinerary = synthesize_itinerary(planner_result["itinerary"])
                            st.session_state.messages.append({"role": "assistant", "content": final_itinerary, "type": "itinerary"})
                        else:
                            st.session_state.messages.append({"role": "assistant", "content": planner_result["error"], "type": "error"})

                    elif agent_type == "research":
                        search_result = web_search(prompt)
                        if "error" in search_result:
                            st.session_state.messages.append({"role": "assistant", "content": search_result["error"], "type": "error"})
                        else:
                            final_answer = run_research_agent(prompt, search_result)
                            st.session_state.messages.append({"role": "assistant", "content": final_answer, "type": "research"})

                    elif agent_type == "finder":
                        finder_result = run_finder_agent(prompt)
                        if "results" in finder_result:
                            st.session_state.messages.append({"role": "assistant", "content": finder_result["results"], "type": "cards"})
                        else:
                            st.session_state.messages.append({"role": "assistant", "content": finder_result.get("error", "An unknown error occurred."), "type": "error"})
            st.rerun()
        else:
            st.error("I need your location to help! Please enable location services in your browser and refresh the page.")