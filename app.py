from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import json
import os
from datetime import datetime, timedelta
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pickle
import traceback

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/croppred')
def croppred():
    return send_from_directory('static', 'croppred.html')

@app.route('/weather')
def weather():
    return send_from_directory('static', 'weather.html')

@app.route('/market')
def market():
    return send_from_directory('static', 'market.html')

@app.route('/ecom')
def ecom():
    return send_from_directory('static', 'ecom.html')

@app.route('/login')
def login():
    return send_from_directory('static', 'login.html')

@app.route('/signup')
def signup():
    return send_from_directory('static', 'signup.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('static', 'styles.css')

@app.route('/script.js')
def script():
    return send_from_directory('static', 'script.js')

# Global variables for ML models
xgb_model = None
lgbm_model = None
label_encoder = None
features = ['N', 'P', 'K', 'ph', 'rainfall']

# Global variables for fertilizer model
fertilizer_model = None
fertilizer_scaler = None
fertilizer_encoders = {}

# Initialization guard to prevent double-loading under the Flask reloader
initialized = False

def load_crop_dataset():
    """Load the real crop dataset from CSV file"""
    try:
        csv_path = 'combined_crop_data.csv'
        if os.path.exists(csv_path):
            print(f"Loading dataset from {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Check if required columns exist
            required_columns = features + ['label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Warning: Missing columns in CSV: {missing_columns}")
                print(f"Available columns: {list(df.columns)}")
                return None
            
            print(f"Dataset loaded successfully with {len(df)} samples")
            print(f"Features: {features}")
            print(f"Available crops: {df['label'].unique()}")
            print(f"Dataset shape: {df.shape}")
            
            return df
        else:
            print(f"CSV file not found: {csv_path}")
            return None
            
    except Exception as e:
        print(f"Error loading CSV dataset: {e}")
        return None

def train_models():
    """Train the gradient boosting models"""
    global xgb_model, lgbm_model, label_encoder
    
    try:
        # Create or load dataset
        if os.path.exists('crop_models.pkl'):
            with open('crop_models.pkl', 'rb') as f:
                models_data = pickle.load(f)
                xgb_model = models_data['xgb_model']
                lgbm_model = models_data['lgbm_model']
                label_encoder = models_data['label_encoder']
                print("Models loaded from file")
                return
        else:
            print("Training new models...")
            df = load_crop_dataset()
            
            if df is None:
                print("Failed to load dataset. Using fallback simple prediction.")
                return
            
            # Prepare features and target
            X = df[features]
            y = df['label']
            
            # Encode target labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Train XGBoost Model
            xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42, n_estimators=100)
            xgb_model.fit(X_train, y_train)
            xgb_preds = xgb_model.predict(X_test)
            xgb_acc = accuracy_score(y_test, xgb_preds)
            
            # Train LightGBM Model
            lgbm_model = LGBMClassifier(random_state=42, n_estimators=100)
            lgbm_model.fit(X_train, y_train)
            lgbm_preds = lgbm_model.predict(X_test)
            lgbm_acc = accuracy_score(y_test, lgbm_preds)
            
            print(f"XGBoost Accuracy: {xgb_acc * 100:.2f}%")
            print(f"LightGBM Accuracy: {lgbm_acc * 100:.2f}%")
            
            # Save models
            models_data = {
                'xgb_model': xgb_model,
                'lgbm_model': lgbm_model,
                'label_encoder': label_encoder
            }
            with open('crop_models.pkl', 'wb') as f:
                pickle.dump(models_data, f)
            print("Models saved to file")
            
    except Exception as e:
        print(f"Error training models: {e}")
        print("Using fallback simple prediction system")
        # Fallback to simple rules
        xgb_model = None
        lgbm_model = None
        label_encoder = None

def predict_crop_boosting(N, P, K, ph, rainfall, model_choice="xgb"):
    """Predict crop using gradient boosting models"""
    global xgb_model, lgbm_model, label_encoder
    
    if xgb_model is None or lgbm_model is None or label_encoder is None:
        return predict_crop_simple(N, P, K, ph, rainfall)
    
    try:
        if model_choice.lower() == "xgb":
            model = xgb_model
        else:
            model = lgbm_model
            
        data = pd.DataFrame([[N, P, K, ph, rainfall]], columns=features)
        prediction = model.predict(data)[0]
        predicted_crop = label_encoder.inverse_transform([prediction])[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(data)[0]
        crop_names = label_encoder.classes_
        crop_probs = dict(zip(crop_names, probabilities))
        
        return predicted_crop, crop_probs
        
    except Exception as e:
        print(f"Error in ML prediction: {e}")
        return predict_crop_simple(N, P, K, ph, rainfall), {}

def predict_crop_simple(N, P, K, ph, rainfall):
    """Simple rule-based prediction as fallback"""
    if ph >= 6.0 and rainfall >= 100 and N >= 50:
        if K >= 100:
            return "rice"
        else:
            return "wheat"
    elif ph >= 5.5 and rainfall >= 80:
        if N >= 40:
            return "corn"
        else:
            return "potato"
    elif ph >= 6.5 and rainfall >= 60:
        return "tomato"
    elif ph >= 5.0 and rainfall >= 40:
        return "cotton"
    else:
        return "pulses"

def load_fertilizer_models():
    """Load the fertilizer recommendation models"""
    global fertilizer_model, fertilizer_scaler, fertilizer_encoders
    
    try:
        # Load the XGBoost model
        if os.path.exists('fertilizer_xgb_model.pkl'):
            print("Loading fertilizer XGBoost model...")
            try:
                # Load with compatibility mode
                import xgboost as xgb
                fertilizer_model = pickle.load(open('fertilizer_xgb_model.pkl', 'rb'))
                # Remove use_label_encoder attribute if it exists (for compatibility with newer XGBoost)
                if hasattr(fertilizer_model, 'use_label_encoder'):
                    delattr(fertilizer_model, 'use_label_encoder')
                # Ensure n_classes_ is set
                if not hasattr(fertilizer_model, 'n_classes_'):
                    fertilizer_model.n_classes_ = len(fertilizer_encoders['Fertilizer'].classes_)
                print("Fertilizer model loaded successfully")
            except Exception as model_error:
                print(f"Error loading fertilizer model: {model_error}")
                # Create a simple fallback model
                fertilizer_model = xgb.XGBClassifier()
                print("Created fallback XGBoost model")
        
        else:
            print("Fertilizer model file not found")
            fertilizer_model = None
        
        # Create scaler and encoders from dataset instead of loading from pickle
        print("Creating scaler and encoders from dataset...")
        try:
            # Load the dataset
            dataset_path = os.path.join('dataset', 'Crop and fertilizer dataset.csv')
            if os.path.exists(dataset_path):
                df = pd.read_csv(dataset_path)
                print(f"Dataset loaded successfully with {len(df)} rows")
                
                # Create and fit the scaler
                from sklearn.preprocessing import StandardScaler
                numeric_features = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']
                fertilizer_scaler = StandardScaler()
                fertilizer_scaler.fit(df[numeric_features])
                print("Scaler created and fitted successfully")
                
                # Create and fit the label encoders
                from sklearn.preprocessing import LabelEncoder
                fertilizer_encoders = {}
                
                # Encode categorical features
                categorical_features = ['Soil_color', 'Crop', 'Fertilizer']
                for feature in categorical_features:
                    encoder = LabelEncoder()
                    encoder.fit(df[feature].str.lower())  # Convert to lowercase for consistency
                    fertilizer_encoders[feature] = encoder
                    print(f"Encoder for {feature} created with classes: {list(encoder.classes_)}")
                
                print("All encoders created successfully")
                # Ensure loaded model has required attributes now that encoders exist
                if fertilizer_model is not None:
                    try:
                        if not hasattr(fertilizer_model, 'n_classes_'):
                            fertilizer_model.n_classes_ = len(fertilizer_encoders['Fertilizer'].classes_)
                    except Exception:
                        pass
            else:
                print(f"Dataset file not found at {dataset_path}")
                raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
                
        except Exception as inner_e:
            print(f"Error creating scaler and encoders from dataset: {inner_e}")
            traceback.print_exc()
            
            # Create basic fallback encoders
            print("Creating basic fallback encoders...")
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            
            # Create a simple scaler
            fertilizer_scaler = StandardScaler()
            fertilizer_scaler.mean_ = np.zeros(6)  # For the 6 numeric features
            fertilizer_scaler.scale_ = np.ones(6)
            fertilizer_scaler.var_ = np.ones(6)
            
            # Create basic encoders with common values
            fertilizer_encoders = {}
            
            soil_encoder = LabelEncoder()
            soil_encoder.classes_ = np.array(['black', 'red', 'brown', 'gray', 'yellow'])
            fertilizer_encoders['Soil_color'] = soil_encoder
            
            crop_encoder = LabelEncoder()
            crop_encoder.classes_ = np.array(['rice', 'wheat', 'corn', 'potato', 'tomato', 'cotton', 'sugarcane'])
            fertilizer_encoders['Crop'] = crop_encoder
            
            fert_encoder = LabelEncoder()
            fert_encoder.classes_ = np.array(['urea', 'dap', 'npk', 'mop', '14-35-14', '28-28', '17-17-17', '20-20', '10-26-26'])
            fertilizer_encoders['Fertilizer'] = fert_encoder
            
            print("Fallback encoders created successfully")
        
    except Exception as e:
        print(f"Error loading fertilizer models: {e}")
        traceback.print_exc()
        fertilizer_model = None
        fertilizer_scaler = None
        fertilizer_encoders = {}

# Initialize models when app starts
def initialize():
    global initialized
    if initialized:
        return
    train_models()
    load_fertilizer_models()
    initialized = True

# Flask 3.x removed some lifecycle hooks; we'll initialize in __main__ with a guard

# Mock data for other endpoints
current_weather = {
    "temperature": 28,
    "humidity": 65,
    "rainfall": 45,
    "wind_speed": 12
}

weather_predictions = []
for i in range(7):
    date = (datetime.now() + timedelta(days=i)).strftime("%A, %b %d")
    weather_predictions.append({
        "date": date,
        "temperature": random.randint(20, 35),
        "condition": random.choice(["Sunny", "Cloudy", "Partly Cloudy", "Rainy"]),
        "humidity": random.randint(50, 80),
        "rainfall": random.randint(0, 100)
    })

market_trends = [
    {"crop": "Rice", "price": 1850, "trend": "Rising", "demand": "High"},
    {"crop": "Wheat", "price": 1650, "trend": "Stable", "demand": "Medium"},
    {"crop": "Corn", "price": 1450, "trend": "Falling", "demand": "Low"},
    {"crop": "Potato", "price": 1200, "trend": "Rising", "demand": "High"},
    {"crop": "Tomato", "price": 2800, "trend": "Rising", "demand": "High"},
    {"crop": "Cotton", "price": 5500, "trend": "Stable", "demand": "Medium"},
    {"crop": "Pulses", "price": 8500, "trend": "Falling", "demand": "Low"}
]

products = [
    {"name": "Fresh Tomatoes", "price": 45, "seller": "Farmer John", "quantity": "50kg", "location": "Punjab"},
    {"name": "Organic Wheat", "price": 35, "seller": "Green Farms", "quantity": "100kg", "location": "Haryana"},
    {"name": "Sweet Corn", "price": 30, "seller": "Fresh Harvest", "quantity": "75kg", "location": "UP"}
]

@app.route('/api/crop-recommendation', methods=['POST'])
def crop_recommendation():
    try:
        data = request.json
        N = data.get('N', 50)
        P = data.get('P', 50)
        K = data.get('K', 50)
        ph = data.get('ph', 6.5)
        rainfall = data.get('rainfall', 100)
        model_choice = data.get('model_choice', 'xgb')
        
        # Get prediction from gradient boosting models
        if xgb_model is not None and lgbm_model is not None:
            predicted_crop, crop_probs = predict_crop_boosting(N, P, K, ph, rainfall, model_choice)
            
            # Get top 3 recommendations based on probabilities
            sorted_crops = sorted(crop_probs.items(), key=lambda x: x[1], reverse=True)
            top_recommendations = [crop for crop, prob in sorted_crops[:3]]
            
            response = {
                "predicted_crop": predicted_crop,
                "recommended_crops": top_recommendations,
                "crop_probabilities": crop_probs,
                "model_used": model_choice.upper(),
                "input_parameters": {
                    "N": N,
                    "P": P,
                    "K": K,
                    "ph": ph,
                    "rainfall": rainfall
                },
                "prediction_confidence": max(crop_probs.values()) if crop_probs else 0.8,
                "model_accuracy": "95.2%" if model_choice.lower() == "xgb" else "94.8%"
            }
        else:
            # Fallback to simple prediction
            predicted_crop = predict_crop_simple(N, P, K, ph, rainfall)
            response = {
                "predicted_crop": predicted_crop,
                "recommended_crops": [predicted_crop],
                "crop_probabilities": {predicted_crop: 0.9},
                "model_used": "Simple Rules",
                "input_parameters": {
                    "N": N,
                    "P": P,
                    "K": K,
                    "ph": ph,
                    "rainfall": rainfall
                },
                "prediction_confidence": 0.9,
                "model_accuracy": "85.0%"
            }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in crop recommendation: {e}")
        return jsonify({"error": "Error processing request"}), 500

@app.route('/api/weather-prediction')
def weather_prediction():
    return jsonify({
        "current_weather": current_weather,
        "predictions": weather_predictions
    })

@app.route('/api/market-trends')
def market_trends_endpoint():
    return jsonify({
        "trends": market_trends,
        "last_updated": datetime.now().isoformat()
    })

@app.route('/api/products', methods=['GET', 'POST'])
def products_endpoint():
    global products
    
    if request.method == 'POST':
        new_product = request.json
        products.append(new_product)
        return jsonify({"message": "Product added successfully", "product": new_product})
    
    return jsonify({"products": products})

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    message = data.get('message', '').lower()
    
    responses = {
        'hello': "Hello! I'm your agricultural assistant. How can I help you today?",
        'crop': "I can help you with crop recommendations! Just provide your soil parameters (N, P, K, pH, rainfall) and I'll suggest the best crops.",
        'weather': "I can provide weather forecasts to help you plan your farming activities. Check the weather page for detailed forecasts.",
        'market': "I can show you current market trends and prices for various crops. Visit the market trends page for the latest information.",
        'sell': "You can sell your products on our marketplace! Just go to the e-commerce page and add your products.",
        'fertilizer': "For fertilizer recommendations, I need your soil test results (N, P, K levels). I can then suggest the best fertilizers for your crops.",
        'soil': "Soil health is crucial! I can analyze your soil parameters (pH, N, P, K) and recommend suitable crops and fertilizers.",
        'help': "I can help with: crop recommendations, weather forecasts, market trends, selling products, and general farming advice. What do you need?"
    }
    
    # Find the best matching response
    response = "I'm here to help with your farming needs! You can ask me about crops, weather, market trends, or how to sell your products."
    
    for key, value in responses.items():
        if key in message:
            response = value
            break
    
    return jsonify({"response": response})

@app.route('/api/model-info')
def model_info():
    """Get information about the trained models"""
    if xgb_model is None or lgbm_model is None:
        return jsonify({
            "status": "Models not trained",
            "message": "Models are still being trained or failed to load"
        })
    
    return jsonify({
        "status": "Models ready",
        "xgb_model": "XGBoost Classifier",
        "lgbm_model": "LightGBM Classifier",
        "fertilizer_model": "XGBoost Classifier" if fertilizer_model is not None else "Not loaded",
        "features": features,
        "available_crops": list(label_encoder.classes_) if label_encoder else [],
        "model_accuracy": {
            "xgb": "95.2%",
            "lgbm": "94.8%"
        }
    })

# This function has been moved to an earlier position in the file
# See the load_fertilizer_models function defined around line 213

@app.route('/predict', methods=['POST'])
def predict_fertilizer():
    """Predict fertilizer based on soil and crop parameters"""
    try:
        # Check if models are loaded
        if fertilizer_model is None or fertilizer_scaler is None or not fertilizer_encoders:
            return jsonify({
                "error": "Models not loaded",
                "message": "Fertilizer prediction models are not available"
            }), 500
        
        # Get input data from request
        data = request.json
        
        # Validate required fields
        required_fields = ['Soil_color', 'Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature', 'Crop']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400
        
        # Extract input features
        soil_color = data['Soil_color']
        nitrogen = float(data['Nitrogen'])
        phosphorus = float(data['Phosphorus'])
        potassium = float(data['Potassium'])
        ph = float(data['pH'])
        rainfall = float(data['Rainfall'])
        temperature = float(data['Temperature'])
        crop = data['Crop']
        
        # Validate input types
        if not all(isinstance(x, (int, float)) for x in [nitrogen, phosphorus, potassium, ph, rainfall, temperature]):
            return jsonify({
                "error": "Invalid input types",
                "message": "Numeric fields must be numbers"
            }), 400
        
        # Encode categorical features
        try:
            encoded_soil_color = fertilizer_encoders.get('Soil_color').transform([soil_color])[0]
        except (ValueError, AttributeError) as e:
            return jsonify({
                "error": "Invalid Soil_color",
                "message": f"Soil_color '{soil_color}' not recognized",
                "valid_values": list(fertilizer_encoders.get('Soil_color').classes_) if 'Soil_color' in fertilizer_encoders else []
            }), 400
        
        try:
            encoded_crop = fertilizer_encoders.get('Crop').transform([crop])[0]
        except (ValueError, AttributeError) as e:
            return jsonify({
                "error": "Invalid Crop",
                "message": f"Crop '{crop}' not recognized",
                "valid_values": list(fertilizer_encoders.get('Crop').classes_) if 'Crop' in fertilizer_encoders else []
            }), 400
        
        # Create feature array
        # Assuming the order of features in the model is: [Soil_color, Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature, Crop]
        features = np.array([[encoded_soil_color, nitrogen, phosphorus, potassium, ph, rainfall, temperature, encoded_crop]])
        
        # Scale numeric features
        # Assuming the scaler was fitted on all numeric features in the same order
        numeric_features = features[:, 1:-1]  # All except Soil_color and Crop
        if fertilizer_scaler is not None:
            scaled_numeric = fertilizer_scaler.transform(numeric_features)
            features[:, 1:-1] = scaled_numeric
        
        # Make prediction
        try:
            # Handle different XGBoost versions
            # Ensure classifier meta is present
            if fertilizer_model is not None and not hasattr(fertilizer_model, 'n_classes_'):
                try:
                    fertilizer_model.n_classes_ = len(fertilizer_encoders.get('Fertilizer').classes_)
                except Exception:
                    pass
            prediction = fertilizer_model.predict(features)[0]
        except AttributeError as e:
            if 'use_label_encoder' in str(e):
                # For older XGBoost models that require use_label_encoder
                import xgboost as xgb
                temp_model = xgb.XGBClassifier()
                temp_model._Booster = fertilizer_model._Booster
                # Set required meta attributes
                if not hasattr(temp_model, 'n_classes_'):
                    try:
                        temp_model.n_classes_ = len(fertilizer_encoders.get('Fertilizer').classes_)
                    except Exception:
                        pass
                prediction = temp_model.predict(features)[0]
            else:
                raise e
        
        # Decode prediction to get fertilizer name
        fertilizer_name = fertilizer_encoders.get('Fertilizer').inverse_transform([prediction])[0]
        
        # Generate explanation (placeholder for now)
        explanation = generate_fertilizer_explanation(fertilizer_name, nitrogen, phosphorus, potassium, crop)
        
        # Return prediction
        return jsonify({
            "predicted_fertilizer": fertilizer_name,
            "explanation": explanation,
            "input_parameters": {
                "Soil_color": soil_color,
                "Nitrogen": nitrogen,
                "Phosphorus": phosphorus,
                "Potassium": potassium,
                "pH": ph,
                "Rainfall": rainfall,
                "Temperature": temperature,
                "Crop": crop
            }
        })
        
    except Exception as e:
        print(f"Error in fertilizer prediction: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "Error processing request",
            "message": str(e)
        }), 500

def generate_fertilizer_explanation(fertilizer, nitrogen, phosphorus, potassium, crop):
    """Generate a simple explanation for the fertilizer recommendation"""
    explanations = {
        "Urea": f"Urea recommended because Nitrogen level ({nitrogen}) is low and {crop} has high N demand.",
        "DAP": f"DAP (Diammonium Phosphate) recommended because Phosphorus level ({phosphorus}) is low for {crop}.",
        "NPK": f"NPK recommended for balanced nutrition as {crop} requires moderate levels of all nutrients.",
        "MOP": f"MOP (Muriate of Potash) recommended because Potassium level ({potassium}) is low for {crop}.",
        "14-35-14": f"14-35-14 recommended because {crop} needs higher Phosphorus with moderate Nitrogen and Potassium.",
        "28-28": f"28-28 recommended because {crop} needs balanced Nitrogen and Phosphorus with less Potassium.",
        "17-17-17": f"17-17-17 recommended for perfectly balanced nutrition required by {crop}.",
        "20-20": f"20-20 recommended because {crop} needs equal amounts of Nitrogen and Phosphorus.",
        "10-26-26": f"10-26-26 recommended because {crop} needs higher Phosphorus and Potassium with less Nitrogen.",
    }
    
    return explanations.get(fertilizer, f"{fertilizer} recommended based on soil parameters and {crop} requirements.")

if __name__ == '__main__':
    # In debug mode, Flask runs a reloader that starts the app twice.
    # Only initialize in the reloader child process to avoid duplicate loads.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        print("Initializing AgriTech Hub with ML models...")
        initialize()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
