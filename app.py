from flask import Flask, render_template, request, send_from_directory, redirect, session, jsonify, url_for
import os
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime
import numpy as np
from openai import OpenAI
import psycopg2
from flask_mail import Mail, Message


load_dotenv()
# Initialize app
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")

mail = Mail(app)

# Database connection

db = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    dbname=os.getenv("DB_NAME"),
    sslmode="require"
)

cursor = db.cursor()


# 🔥 ADD YOUR API KEY HERE
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load model
model = load_model("model.h5")

# Upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Class labels
classes = [
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Yellow Leaf Curl Virus",
    "Healthy"
]

# Disease full info
disease_info = {

    "Late Blight": {
        "description": "Late Blight is a fast-spreading fungal disease that produces dark, water-soaked lesions on leaves and stems. It can destroy an entire tomato crop under cool and humid conditions.",

        "causes": [
            "Cool and humid climate",
            "Continuous rainfall",
            "High moisture on leaves",
            "Infected plant material"
        ],

        "treatment": [
            "Remove infected plant parts",
            "Spray an approved fungicide",
            "Improve field drainage",
            "Monitor plants regularly"
        ],

        "prevention": [
            "Avoid overhead irrigation",
            "Ensure proper airflow",
            "Use disease-free seedlings",
            "Inspect crops frequently"
        ]
    },


    "Early Blight": {
        "description": "Early Blight is a common fungal disease that causes dark brown spots with concentric rings on older tomato leaves. If untreated, it spreads and reduces plant growth and fruit yield.",

        "causes": [
            "Warm and humid weather",
            "Fungal spores in soil",
            "Overhead watering",
            "Poor air circulation"
        ],

        "treatment": [
            "Remove infected leaves immediately",
            "Apply a recommended fungicide",
            "Avoid wetting the foliage",
            "Dispose of infected plant debris"
        ],

        "prevention": [
            "Practice crop rotation",
            "Maintain proper plant spacing",
            "Water near the roots only",
            "Keep the field free from weeds"
        ]
    },

    "Leaf Mold": {
        "description": "Leaf Mold is a fungal disease that mainly affects tomato leaves in humid environments. Yellow spots appear on the upper surface while mold develops underneath.",

        "causes": [
            "High humidity",
            "Poor ventilation",
            "Wet leaves for long periods",
            "Dense plant spacing"
        ],

        "treatment": [
            "Remove affected leaves",
            "Increase air circulation",
            "Apply suitable fungicide",
            "Reduce greenhouse humidity"
        ],

        "prevention": [
            "Maintain proper spacing",
            "Improve ventilation",
            "Avoid excessive watering",
            "Keep foliage dry"
        ]
    },

    "Yellow Leaf Curl Virus": {
        "description": "Yellow Leaf Curl Virus is a serious viral disease that causes yellowing, curling, and stunted growth in tomato plants. It is mainly spread by whiteflies.",

        "causes": [
            "Whitefly infestation",
            "Infected tomato plants",
            "Warm weather",
            "Poor pest management"
        ],

        "treatment": [
            "Remove infected plants",
            "Control whiteflies promptly",
            "Use insect traps",
            "Monitor nearby plants"
        ],

        "prevention": [
            "Plant resistant varieties",
            "Control whitefly population",
            "Keep the field weed-free",
            "Inspect crops regularly"
        ]
    },

    "Healthy": {
        "description": "The uploaded tomato leaf appears healthy with no visible signs of disease. Continue following proper crop management practices to maintain healthy plant growth.",

        "causes": [
            "Proper irrigation",
            "Balanced nutrition",
            "Good field hygiene",
            "Suitable weather conditions"
        ],

        "treatment": [
            "No treatment is required",
            "Continue regular monitoring",
            "Maintain balanced fertilization",
            "Follow routine crop care"
        ],

        "prevention": [
            "Inspect plants regularly",
            "Maintain proper watering",
            "Control pests early",
            "Ensure balanced soil nutrition"
        ]
    },
}

# 🔥 AI CHATBOT LOGIC
from difflib import get_close_matches

def chatbot_logic(user_message):
    msg = user_message.lower().strip()
    last_disease = session.get('last_disease')

    # 🟢 1. ML CONTEXT (Image prediction based)
    if "this disease" in msg and last_disease:
        info = disease_info.get(last_disease, {})
        return f"""Disease: {last_disease}

Description: {info.get('description', '')}

Treatment: {', '.join(info.get('treatment', []))}

Prevention: {', '.join(info.get('prevention', []))}"""

    # 🟢 2. QUESTION-ANSWER DATASET (UNIQUE KEYS)
    qa_pairs = {

        "hello": "Hello! How can I help you today? You can ask me about plant diseases, treatments, or farming tips.",

        "what is late blight": "Late blight is a serious fungal disease that affects crops like potato and tomato. It causes dark spots on leaves and spreads quickly in humid conditions.",

        "what is early blight": "Early blight is a common disease that causes brown spots with rings on leaves, mainly affecting tomato and potato plants.",

        "what is leaf mold": "Leaf mold is a fungal disease that appears as yellow spots on leaves and mold growth underneath in humid conditions.",

        "what is curl virus": "Curl virus is a viral disease that causes leaves to curl, shrink, and deform. It spreads through insects like whiteflies.",

        "what causes leaf spots on plants": "Leaf spots are caused by fungi, bacteria, or environmental conditions like excess moisture and poor airflow.",

        "why are my plant leaves turning yellow": "Leaves turn yellow due to nutrient deficiency, overwatering, poor drainage, or disease.",

        "why are my leaves drying": "Leaves dry due to lack of water, heat stress, disease, or nutrient imbalance.",

        "what disease affects tomato plants": "Tomato plants are affected by diseases like early blight, late blight, leaf mold, and curl virus.",

        "my plant has white spots, what is it": "White spots may indicate fungal infections like powdery mildew or leaf mold.",

        "my leaves have brown patches, why": "Brown patches are usually caused by early blight, sunburn, or nutrient deficiency.",

        "leaves are curling, what is the reason": "Leaf curling is caused by viral infections, pests, or environmental stress.",

        "there are holes in leaves, why": "Holes in leaves are caused by insects like caterpillars and beetles.",

        "how to treat late blight": "Use fungicides, remove infected leaves, and avoid excess moisture.",

        "how to treat early blight": "Remove affected leaves, apply fungicide, and follow crop rotation.",

        "how to treat leaf mold": "Improve airflow, reduce humidity, and use fungicides.",

        "how to treat curl virus": "Remove infected plants and control whiteflies. There is no direct cure.",

        "how to cure plant diseases naturally": "Use neem oil, baking soda spray, and maintain proper watering.",

        "which pesticide should i use": "Use pesticides based on the issue. Neem oil is a safe general option.",

        "how to stop fungal infection in crops": "Avoid overwatering, improve air circulation, and use fungicides.",

        "what cause late blight": "Late blight is caused by a fungus-like organism in cool and humid conditions.",

        "what cause early blight": "Early blight is caused by fungal pathogens in warm and humid environments.",

        "what cause leaf mold": "Leaf mold is caused by fungi that grow in high humidity and poor ventilation.",

        "what cause curl virus": "Curl virus is caused by viruses spread by whiteflies.",

        "how to prevent plant diseases": "Use healthy seeds, proper spacing, and avoid excess moisture.",

        "how to protect crops from pests": "Use natural pesticides, monitor crops, and remove infected parts.",

        "how to maintain healthy soil": "Use compost, crop rotation, and maintain soil nutrients.",

        "what is ideal soil ph for crops": "Most crops grow well in soil pH between 6.0 and 7.5.",

        "how does weather affect crops": "Weather affects crop growth. Too much rain, heat, or cold can damage plants."
    }

    # 🟢 3. EXACT MATCH (BEST ACCURACY)
    if msg in qa_pairs:
        return qa_pairs[msg]

    # 🟡 4. FUZZY MATCH (HANDLE TYPOS)
    closest = get_close_matches(msg, qa_pairs.keys(), n=1, cutoff=0.6)
    if closest:
        return qa_pairs[closest[0]]

    # 🔵 5. AI FALLBACK (OPTIONAL BUT POWERFUL)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an agriculture expert helping farmers."},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content

    except:
        return "Sorry, I didn't understand. Please ask a question related to plant diseases or farming."

# Welcome Page
@app.route('/')
def welcome():
    return render_template('welcome.html')

# Store username,state and district in session
@app.route('/setname', methods=['POST'])
def setname():
    session['username'] = request.form['username']
    session['state'] = request.form['state']
    session['district'] = request.form.get('district', '')
    return redirect('/home')

# Main Page
@app.route('/home')
def home():

    return render_template(
        "index.html",
        username=session.get("username", "Farmer"),
        state=session.get("state"),
        district=session.get("district")
    )

# 🔥 Predict route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    username = session.get('username', 'Farmer')

    if file and file.filename != '':
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]

        best_idx = np.argmax(predictions)
        best_class = classes[best_idx]
        best_conf = float(round(predictions[best_idx] * 100, 2))
        session['last_disease'] = best_class

        print("Username:", session.get("username"))
        print("State:", session.get("state"))

        # Get user details
        username = session.get("username", "Farmer")
        state = session.get("state", "Unknown")

        # Save prediction into MySQL
        sql = """
        INSERT INTO prediction_history
        (username, state, image_path, disease_name, confidence)
        VALUES (%s, %s, %s, %s, %s)
        """

        values = (
            username,
            state,
            file.filename,
            best_class,
            best_conf
        )       

        cursor.execute(sql, values)
        db.commit()


        info_data = disease_info.get(best_class, {})

        if best_class == "Healthy":
            severity = "Healthy"
        elif best_conf >= 85:
            severity = "Severe"
        elif best_conf >= 60:
            severity = "Moderate"
        else:
            severity = "Mild"

        image_path = '/uploads/' + file.filename

    else:
        image_path = None
        best_class = "No Image"
        best_conf = 0
        severity = "N/A"
        info_data = {
            "description": "-",
            "causes": [],
            "treatment": [],
            "prevention": []
        }

    return render_template(
        'result.html',
        image_path=image_path,
        disease=best_class,
        confidence=best_conf,
        info=info_data,
        severity=severity,
        username=username
    )

# 🔥 CHATBOT API ROUTE
@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.json["message"]
    reply = chatbot_logic(user_message)
    return jsonify({"reply": reply})

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 🔥 TEST (OPTIONAL - REMOVE LATER)
# print(chatbot_logic("What is tomato late blight?"))


@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")


@app.route("/about")
def about():
    return render_template("about.html")


# ---------------- WEATHER ----------------
# ---------------- WEATHER ----------------

import requests

@app.route('/weather')
def weather():

    api_key = os.getenv("WEATHER_API_KEY")

    # ✅ GET PARAMETERS
    city = request.args.get("city")
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    # ✅ PRIORITY: CITY FIRST
    if city:
        city = city.strip()
        print("City:", city)

        if not city.replace(" ", "").isalpha():
            return render_template('weather.html', error="Please enter a valid city name")

        url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"

    # ✅ OTHERWISE USE GPS
    elif lat and lon:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"

    # ✅ NO INPUT
    else:
        return render_template('weather.html')

    # 🔥 API CALL
    response = requests.get(url)
    data = response.json()
    print(data)

    # ❌ ERROR CHECK
    if data.get("cod") != "200":
        return render_template('weather.html', error=data.get("message"))

    # ✅ CURRENT WEATHER
    current = data['list'][0]

    weather_data = {
        "city": data['city']['name'],
        "temp": round(current['main']['temp'], 1),
        "humidity": current['main']['humidity'],
        "description": current['weather'][0]['description'],
        "icon": current['weather'][0]['icon']
    }

    # ✅ FORECAST

    # ✅ DAILY FORECAST (NEXT 5 DAYS)

    forecast_list = []
    today = datetime.now().date()
    days_added = []
    for item in data['list']:
        dt = datetime.strptime(item['dt_txt'],
                           "%Y-%m-%d %H:%M:%S")

        # Skip today's date
        if dt.date() == today:
            continue

        day_name = dt.strftime("%A")

        # Prefer forecast around noon
        hour = dt.hour

        if 11 <= hour <= 14 and day_name not in days_added:

            forecast_list.append({
                "time": day_name,
                "temp": round(item['main']['temp'],1),
                "desc": item['weather'][0]['description'],
                "icon": item['weather'][0]['icon']
            })

            days_added.append(day_name)

        if len(forecast_list) == 5:
            break


    # 🌧️ RAIN CHECK
    rain_coming = any(
        "rain" in item['weather'][0]['description'].lower()
        for item in data['list']
    )

    # 🌱 FARMING ADVICE
    advice = []

    # Temperature
    if weather_data['temp'] > 35:
        advice.append("High temperature: Irrigate crops regularly 🌡️💧")
    elif weather_data['temp'] < 15:
        advice.append("Low temperature: Protect crops from cold/frost ❄️")

    # Humidity
    if weather_data['humidity'] > 80:
        advice.append("High humidity: Risk of fungal diseases 🌱⚠️")
    elif weather_data['humidity'] < 30:
        advice.append("Low humidity: Soil may dry quickly, increase watering 💧")

    # Condition
    desc = weather_data['description'].lower()

    if rain_coming or "rain" in desc:
        advice.append("Rain expected soon: Avoid spraying pesticides ☔")
    elif "clear" in desc:
        advice.append("Clear weather: Good for farming activities ☀️")
    elif "cloud" in desc:
        advice.append("Cloudy weather: Monitor crops for pests 🌥️")
    elif "haze" in desc or "mist" in desc:
        advice.append("Low visibility: Monitor crop health 🌫️")

    # Default advice
    if not advice:
        advice.append("Weather is normal: Continue regular farming activities 🌱")

    # Extra advice
    if rain_coming:
        advice.append("Ensure proper drainage to avoid waterlogging 🌧️")
    else:
        advice.append("Check soil moisture regularly 🌱")
        advice.append("Ensure proper irrigation schedule 💧")

    return render_template(
        'weather.html',
        weather=weather_data,
        advice=advice,
        forecast=forecast_list
    )

# ---------------- Krishinetra Intro ----------------
# ---------------- Krishinetra Intro ----------------

@app.route("/project-details")
def project_details():
    return render_template("project-details.html")


# ---------------- Contact Page ----------------

@app.route('/contact', methods=['GET', 'POST'])
def contact():

    if request.method == "GET":
        return render_template("contact.html")

    try:
        name = request.form["name"]
        email = request.form["email"]
        subject = request.form["subject"]
        message = request.form["message"]

        msg = Message(
            subject=f"New Contact Form: {subject}",
            sender=app.config['MAIL_USERNAME'],
            recipients=[app.config['MAIL_USERNAME']],
            reply_to=email
        )

        msg.body = f"""
New Message from KrishiNetra

Name : {name}
Email : {email}
Subject : {subject}

Message:

{message}
"""

        mail.send(msg)

        # Auto-reply to the user
        auto_reply = Message(
            subject="Thank you for contacting KrishiNetra",
            sender=app.config['MAIL_USERNAME'],
            recipients=[email]
        )

        auto_reply.body = f"""
🌱 KrishiNetra – AI Powered Agriculture Assistant

Hello {name},

Thank you for contacting KrishiNetra!

We have successfully received your message.

I appreciate your interest and will review your inquiry as soon as possible. I'll get back to you at the earliest opportunity.


--------------------------------------------------

📌 Your Submitted Details

Name    : {name}
Email   : {email}
Subject : {subject}

--------------------------------------------------

Thank you for your interest in KrishiNetra.

Have a wonderful day! 🌿

Best Regards,

Keshri Nandan
Developer - KrishiNetra
📧 {app.config['MAIL_USERNAME']}
"""

        mail.send(auto_reply)

        return jsonify({
            "success": True,
            "message": "Your message has been sent successfully!"
        })

    except Exception as e:
        print("Contact Form Error:", e)

        return jsonify({
            "success": False,
            "message": "Failed to send message."
        })
    
# ---------------- working Page ----------------
@app.route('/working')
def working():
    return render_template('working.html')

# ---------------- disease library Page ----------------
@app.route('/disease-library')
def disease_library():
    return render_template('disease_library.html')


# ---------------- history Page ----------------
@app.route('/history')
def history():

    username = session.get('username')
    filter_type = request.args.get("filter", "all")

    if not username:
        return redirect('/')

# ================= FILTER =================

    if filter_type == "week":

        sql = """
        SELECT id,
            image_path,
            disease_name,
            confidence,
            state,
            prediction_time
        FROM prediction_history
        WHERE username=%s
        AND prediction_time >= NOW() - INTERVAL '7 days'
        ORDER BY prediction_time DESC
        """

        cursor.execute(sql, (username,))

    elif filter_type == "month":

        sql = """
        SELECT id,
            image_path,
            disease_name,
            confidence,
            state,
            prediction_time
        FROM prediction_history
        WHERE username=%s
        AND prediction_time >= NOW() - INTERVAL '30 days'
        ORDER BY prediction_time DESC
        """

        cursor.execute(sql, (username,))

    else:

        sql = """
        SELECT id,
            image_path,
            disease_name,
            confidence,
            state,
            prediction_time
        FROM prediction_history
        WHERE username=%s
        ORDER BY prediction_time DESC
        """

        cursor.execute(sql, (username,))
    
    history = cursor.fetchall()

    # ================= Statistics =================

    total_predictions = len(history)

    healthy_count = sum(
        1 for row in history
        if row[2].lower() == "healthy"
    )

    diseased_count = total_predictions - healthy_count

    if total_predictions > 0:
        average_confidence = round(
            sum(row[3] for row in history) / total_predictions,
            2
        )
    else:
        average_confidence = 0


    return render_template(
        "history.html",

        history=history,

        username=session.get("username"),
        state=session.get("state"),
        district=session.get("district"),
        filter_type=filter_type,

        total_predictions=total_predictions,
        healthy_count=healthy_count,
        diseased_count=diseased_count,
        average_confidence=average_confidence
    )

# ---------------- print history Page ----------------

@app.route("/history/print")
def print_history():

    username = session.get("username")

    if not username:
        return redirect("/")

    sql = """
    SELECT image_path,
           disease_name,
           confidence,
           state,
           prediction_time
    FROM prediction_history
    WHERE username=%s
    ORDER BY prediction_time DESC
    """

    cursor.execute(sql, (username,))
    history = cursor.fetchall()

    total_predictions = len(history)

    healthy_count = sum(1 for row in history if row[1] == "Healthy")

    diseased_count = total_predictions - healthy_count

    average_confidence = round(
        sum(row[2] for row in history) / total_predictions, 2
    ) if total_predictions else 0

    current_date = datetime.now()

    return render_template(
        "print_history.html",
        history=history,
        username=session.get("username"),
        state=session.get("state"),
        district=session.get("district"),
        total_predictions=total_predictions,
        healthy_count=healthy_count,
        diseased_count=diseased_count,
        average_confidence=average_confidence,
        current_date=current_date
    )

@app.route("/delete_history/<int:id>")
def delete_history(id):

    if "username" not in session:
        return redirect("/")

    sql = """
    DELETE FROM prediction_history
    WHERE id=%s
    AND username=%s
    """

    cursor.execute(sql, (id, session["username"]))
    db.commit()

    return redirect(url_for("history"))


@app.route("/logout")
def logout():

    print("Logout route called")

    session.clear()

    return redirect("/")

def create_tables():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME"),
        sslmode="require"
    )

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255),
            image_path VARCHAR(500),
            disease_name VARCHAR(255),
            confidence FLOAT,
            state VARCHAR(255),
            prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()


create_tables()

# Run app
if __name__ == '__main__':
    app.run(debug=True)


    