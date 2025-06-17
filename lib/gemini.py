from google import genai
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

uploaded_file = client.files.upload(file="파일명.json")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["Give me the feedback of this speech", uploaded_file]
)


print(response.text)