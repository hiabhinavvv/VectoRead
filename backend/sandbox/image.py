import ollama

stream = ollama.chat(
    model="llava:7b",
    messages=[{
        'role':'user',
        'content':'can you transcribe the text from this image?',
        'images': ["data/image.jpeg"]
    }],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)