from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from difflib import SequenceMatcher

app = Flask(__name__)

df = pd.read_csv("soru_cevap_veri.csv", encoding="windows-1254", sep=None, engine="python", on_bad_lines='skip')
df.columns = ["soru", "cevap"]
df = df.dropna()  # ✅ Boş hücreleri temizle

questions = df["soru"]
answers = df["cevap"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)




# 3. Cevap Bulma
def get_bot_response(user_input):
    input_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(input_vec, X).flatten()
    best_match = similarities.argmax()
    if similarities[best_match] < 0.3:
        return "Üzgünüm, bu konuda bir bilgim yok. Daha sonra tekrar sorun."
    return answers[best_match]

# 4. Cevap Benzerliği
def is_similar(a, b, threshold=0.7):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


# 7. HTML Arayüz (sade)
HTML_PAGE = '''
<!doctype html>
<html>
  <head>
    <title>İzüBot</title>
    <style>
      body { font-family: Arial; background: #f0f2f5; padding: 20px; }
      .chatbox { background: white; border-radius: 10px; padding: 20px; max-width: 600px; margin: auto; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
      .msg { margin: 10px 0; }
      .user { text-align: right; }
      .bot { text-align: left; color: #0056b3; }
      input, button { padding: 10px; width: 80%; margin-top: 10px; border-radius: 5px; border: 1px solid #ccc; }
    </style>
  </head>
  <body>
    <div class="chatbox">
      <h2>İzüBot 🤖</h2>
      <div id="chatlog"></div>
      <input type="text" id="userInput" placeholder="Bir şeyler yaz..." />
      <button onclick="sendMessage()">Gönder</button>
    </div>
    <script>
      async function sendMessage() {
        const input = document.getElementById("userInput");
        const message = input.value;
        if (!message) return;
        const chatlog = document.getElementById("chatlog");
        chatlog.innerHTML += '<div class="msg user"><strong>Sen:</strong> ' + message + '</div>';
        input.value = "";

        const response = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: message })
        });
        const data = await response.json();
        chatlog.innerHTML += '<div class="msg bot"><strong>İzüBot:</strong> ' + data.response + '</div>';
      }
    </script>
  </body>
</html>
'''

# 8. Ana Sayfa
@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

# 9. Chat API
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    response = get_bot_response(user_input)
    return jsonify({"response": response})

# 10. Başlat + grafiği göster
if __name__ == "__main__":
    app.run(debug=True, port=5050)
