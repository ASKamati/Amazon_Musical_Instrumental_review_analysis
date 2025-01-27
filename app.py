from flask import Flask, request, jsonify, render_template
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    # Serve the HTML frontend
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        input_data = request.json
        review_text = input_data.get('reviewText', '')

        # Check if input is valid
        if not review_text:
            return jsonify({'error': 'Invalid input. "reviewText" field is required.'}), 400

        # Preprocess and predict
        vectorized_text = vectorizer.transform([review_text])
        prediction = model.predict(vectorized_text)[0]

        # Return prediction
        return jsonify({'reviewText': review_text, 'sentiment': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app on localhost (127.0.0.1) with debug mode enabled
    app.run(host='127.0.0.1', port=5000, debug=True)
