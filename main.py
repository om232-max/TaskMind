import streamlit as st
import os
import datetime
from json import load, dump
from googlesearch import search
from groq import Groq
import google.generativeai as genai

Username = "Broddy"
Assistantname = "TaskMind"

GroqAPIkey = st.secrets["GroqAPIkey"]
GeminiAPIkey = st.secrets["GeminiAPIkey"]

client = Groq(api_key=GroqAPIkey)
genai.configure(api_key=GeminiAPIkey)
model = genai.GenerativeModel("gemini-1.5-pro")

# Models and decision-making prompt
funcs = ["exit", "general", "realtime", "open", "close", "play", "generate image",
         "content", "google search", "youtube search", "reminder", "map"]

preamble = """
You are a very accurate Decision-Making Model, which decides what kind of a query is given to you.
You will decide whether a query is a 'general' query, a 'realtime' query, or is asking to perform any task or automation like 'open facebook, instagram', 'can you write a application and open it in notepad'
*** Do not answer any query, just decide what kind of query is given to you. ***
-> Respond with 'general ( query )' if a query can be answered by a llm model (conversational ai chatbot) and doesn't require any up to date information like if the query is 'who was akbar?' respond with 'general who was akbar?', if the query is 'how can i study more effectively?' respond with 'general how can i study more effectively?', if the query is 'can you help me with this math problem?' respond with 'general can you help me with this math problem?', if the query is 'Thanks, i really liked it.' respond with 'general thanks, i really liked it.' , if the query is 'what is python programming language?' respond with 'general what is python programming language?', etc. Respond with 'general (query)' if a query doesn't have a proper noun or is incomplete like if the query is 'who is he?' respond with 'general who is he?', if the query is 'what's his networth?' respond with 'general what's his networth?', if the query is 'tell me more about him.' respond with 'general tell me more about him.', and so on even if it require up-to-date information to answer. Respond with 'general (query)' if the query is asking about time, day, date, month, year, etc like if the query is 'what's the time?' respond with 'general what's the time?'.
-> Respond with 'realtime ( query )' if a query can not be answered by a llm model (because they don't have realtime data) and requires up to date information like if the query is 'who is indian prime minister' respond with 'realtime who is indian prime minister', if the query is 'tell me about facebook's recent update.' respond with 'realtime tell me about facebook's recent update.', if the query is 'tell me news about coronavirus.' respond with 'realtime tell me news about coronavirus.', etc and if the query is asking about any individual or thing like if the query is 'who is akshay kumar' respond with 'realtime who is akshay kumar', if the query is 'what is today's news?' respond with 'realtime what is today's news?', if the query is 'what is today's headline?' respond with 'realtime what is today's headline?', etc.
-> Respond with 'open (application name or website name)' if a query is asking to open any application like 'open facebook', 'open telegram', etc. but if the query is asking to open multiple applications, respond with 'open 1st application name, open 2nd application name' and so on and correct the application name only if application is Instagram,Whatsapp or youtbe.
-> Respond with 'map (location)' if a query is asking to open the map of a specific location
-> Respond with 'close (application name)' if a query is asking to close any application like 'close notepad', 'close facebook', etc. but if the query is asking to close multiple applications or websites, respond with 'close 1st application name, close 2nd application name' and so on.
-> Respond with 'play (song name)' if a query is asking to play any song like 'play afsanay by ys', 'play let her go', etc. but if the query is asking to play multiple songs, respond with 'play 1st song name, play 2nd song name' and so on.
-> Respond with 'generate image (image prompt)' if a query is requesting to generate a image with given prompt like 'generate image of a lion', 'generate image of a cat', etc. but if the query is asking to generate multiple images, respond with 'generate image 1st image prompt, generate image 2nd image prompt' and so on.
-> Respond with 'reminder (datetime with message)' if a query is requesting to set a reminder like 'set a reminder at 9:00pm on 25th june for my business meeting.' respond with 'reminder 9:00pm 25th june business meeting'.
-> Respond with 'content (topic)' if a query is asking to write any type of content like application, codes, emails or anything else about a specific topic but if the query is asking to write multiple types of content, respond with 'content 1st topic, content 2nd topic' and so on.
-> Respond with 'google search (topic)' if a query is asking to search a specific topic on google but if the query is asking to search multiple topics on google, respond with 'google search 1st topic, google search 2nd topic' and so on.
-> Respond with 'youtube search (topic)' if a query is asking to search a specific topic on youtube but if the query is asking to search multiple topics on youtube, respond with 'youtube search 1st topic, youtube search 2nd topic' and so on.
*** If the query is asking to perform multiple tasks like 'open facebook, telegram and close whatsapp' respond with 'open facebook, open telegram, close whatsapp' ***
*** If the user is saying goodbye or wants to end the conversation like 'bye jarvis.' respond with 'exit'.***
*** Respond with 'general (query)' if you can't decide the kind of query or if a query is asking to perform a task which is not mentioned above. ***
"""


def FirstLayerDMM(preamble, prompt):
    full_prompt = f"{preamble}\n\n{prompt}"
    response = model.generate_content(full_prompt)
    output = response.text.strip()
    phrases = [phrase.strip() for phrase in output.split(",")]
    return [phrase for phrase in phrases if any(phrase.startswith(f) for f in funcs)]


# Utility functions
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
def results(user, prompt):
    # Step 1: Run Decision-Making Model
    decisions = FirstLayerDMM(preamble, prompt)
    if not decisions:
        return "Sorry, I couldn't understand your query."

    # Identify task
    task = decisions[0]

    # Step 2: Build system prompt
    system_prompt = f"""Hello, I am {user}, You are a very accurate and advanced AI chatbot named {Assistantname} which also has real-time up-to-date information from the internet.
    *** Do not tell time until I ask, do not talk too much, just answer the question.***
    *** Reply in only English, even if the question is in Hindi, reply in English.***
    *** Do not provide notes in the output, just answer the question and never mention your training data. ***"""

    # Step 3: Route and Generate Final Response
    if task.startswith("general"):
        answer = ChatBot(user, prompt, system_prompt)

    elif task.startswith("realtime"):
        search_data = GoogleSearch(prompt)
        answer = ChatBot(user, prompt + "\n\n" + search_data, system_prompt)

    else:
        answer = f"Task not supported yet: {task}"

    return answer


