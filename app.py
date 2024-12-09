from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Memuat model dan vectorizer dari file
try:
    model = joblib.load(os.path.join('model', 'model.pkl'))
    vectorizer = joblib.load(os.path.join('model', 'tfidf.pkl'))
except Exception as e:
    raise RuntimeError(f"Error loading model or tfidf: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '')  # Mengambil teks dari form input
    
    if not text.strip():
        return render_template('result.html', prediction="Input teks kosong. Silakan masukkan teks.")
    
    try:
        # Mengubah teks menjadi fitur numerik menggunakan vectorizer
        text_input = vectorizer.transform([text])
        
        # Proses prediksi dengan model
        prediction = model.predict(text_input)[0]
        
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return render_template('result.html', prediction=f"Terjadi error selama prediksi: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5005)
