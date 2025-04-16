import os
import datetime
import asyncio
import time
import re
from random import randint
from json import load, dump
from googlesearch import search
from groq import Groq
import google.generativeai as genai
import requests
import streamlit as st

# API keys from secrets
HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
GroqAPIkey = st.secrets["GROQ_API_KEY"]
GeminiAPIkey = st.secrets["GEMINI_API_KEY"]

# Create image output directory
output_folder = "generated_images"
os.makedirs(output_folder, exist_ok=True)

Username = "Broddy"
Assistantname = "TaskMind"

# Configure models
client = Groq(api_key=GroqAPIkey)
genai.configure(api_key=GeminiAPIkey)
model = genai.GenerativeModel("gemini-1.5-pro")

# Supported tasks
funcs = ["exit", "general", "realtime", "open", "close", "play", "generate image",
         "content", "google search", "youtube search", "reminder", "map"]

# Classifier prompt
preamble = """
You are a highly accurate Decision-Making Model.
Given a user query, classify it into one or more of the following task types:
exit, general, realtime, open, close, play, generate image, content, google search, youtube search, reminder, map.
Respond with the most relevant task label(s) separated by commas, no explanation.

Examples:
Input: "What's the weather like today?" Output: realtime
Input: "Draw a castle in the sky" Output: generate image
Input: "Explain black holes" Output: general
"""

def FirstLayerDMM(preamble, prompt):
    full_prompt = f"{preamble}\n\n{prompt}"
    response = model.generate_content(full_prompt)
    output = response.text.strip()
    phrases = [phrase.strip() for phrase in output.split(",")]
    return [phrase for phrase in phrases if any(phrase.startswith(f) for f in funcs)]

def RealtimeInformation():
    return datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")

def GoogleSearch(query):
    results = list(search(query, advanced=True, num_results=5))
    Answer = f"The search results for '{query}' are:\n[start]\n"
    for i in results:
        Answer += f"Title: {i.title}\nDescription: {i.description}\n\n"
    Answer += "[end]"
    return Answer

def AnswerModifier(answer):
    return '\n'.join([line for line in answer.split('\n') if line.strip()]).replace("</s>", "")

def ChatBot(user, query, system_prompt):
    log_path = f"chat_logs/{user}.json"
    os.makedirs("chat_logs", exist_ok=True)
    try:
        with open(log_path, "r") as f:
            messages = load(f)
    except FileNotFoundError:
        messages = []

    system_messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": query})

    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=system_messages + [{"role": "system", "content": RealtimeInformation()}] + messages,
        max_tokens=1024,
        temperature=0.7,
        top_p=1,
        stream=True
    )

    Answer = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            Answer += chunk.choices[0].delta.content

    messages.append({"role": "assistant", "content": Answer})

    with open(log_path, "w") as f:
        dump(messages, f, indent=4)

    return AnswerModifier(Answer)

async def query_image(payload):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    response = await asyncio.to_thread(requests.post,
        "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
        headers=headers,
        json=payload
    )
    if response.status_code == 200:
        return response.content
    return None

def safe_filename(prompt):
    safe_prompt = re.sub(r'[^a-zA-Z0-9_]', '_', prompt)[:50]
    return f"{safe_prompt}_{int(time.time())}.jpg"

async def generate_image(prompt: str):
    payload = {
        "inputs": f"{prompt}, quality=4K, sharpness=maximum, Ultra High details, high resolution, seed={randint(0, 1000000)}"
    }
    image_bytes = await query_image(payload)
    if image_bytes:
        filename = safe_filename(prompt)
        file_path = os.path.join(output_folder, filename)
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        return file_path
    return None

def results(user, prompt):
    decisions = FirstLayerDMM(preamble, prompt)
    if not decisions:
        return {"type": "text", "content": "Sorry, I couldn't understand your query."}

    task = decisions[0]

    system_prompt = f"""Hello, I am {user}, You are a very accurate and advanced AI chatbot named {Assistantname}.
    *** Do not tell time unless asked. ***
    *** Always reply in English even if the input is in another language. ***
    *** Avoid excessive commentary and disclaimers. ***"""

    if task.startswith("general"):
        text = ChatBot(user, prompt, system_prompt)
        return {"type": "text", "content": text}

    elif task.startswith("realtime"):
        search_data = GoogleSearch(prompt)
        text = ChatBot(user, prompt + "\n\n" + search_data, system_prompt)
        return {"type": "text", "content": text}

    elif task.startswith("generate image"):
        prompt_text = prompt.strip().replace("generate image", "").strip()
        path = asyncio.run(generate_image(prompt_text))
        if path:
            return {"type": "image", "path": path}
        else:
            return {"type": "text", "content": "Image generation failed. Please try again."}

    return {"type": "text", "content": f"Task not supported yet: {task}"}
