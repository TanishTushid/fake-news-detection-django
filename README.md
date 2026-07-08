# 📰 Fake News Detection using Django & Machine Learning

An end-to-end **Fake News Detection** web application built with **Django**, **Scikit-learn**, and **Natural Language Processing (NLP)**. The application predicts whether a news article is **Real** or **Fake** using a trained machine learning model.

---

## 🚀 Features

* 🔍 Predicts whether a news article is **Real** or **Fake**
* 🤖 Machine Learning model trained on a news dataset
* 📝 User-friendly web interface built with Django
* ⚡ Real-time predictions
* 💾 Pre-trained model and vectorizer saved using Pickle
* 📱 Responsive UI

---

## 🛠️ Tech Stack

### Backend

* Python
* Django

### Machine Learning

* Scikit-learn
* Pandas
* NumPy

### NLP

* TF-IDF Vectorizer

### Frontend

* HTML
* CSS
* Bootstrap
* JavaScript

---

## 📂 Project Structure

```text
FakeNewsDetection/
│
├── core/
│
├── detector/
│   ├── migrations/
│   ├── templates/
│   │      home.html
│   │      result.html
│   │      history.html
│   │      about.html
│   │
│   ├── static/
│   │      css/
│   │      images/
│   │
│   ├── ml/
│   │      model.pkl
│   │      vectorizer.pkl
│   │
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   └── utils.py
│
├── dataset/
│      Fake.csv
│      True.csv
│
├── train_model.py
├── requirements.txt
├── manage.py
└── README.md

```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/TanishTushid/fake-news-detection-django.git
```

### 2. Move into the project

```bash
cd fake-news-detection-django
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the server

```bash
python manage.py runserver
```

Open your browser and visit:

```text
http://127.0.0.1:8000/
```

---

## 📊 Machine Learning Pipeline

1. Data Collection
2. Data Cleaning
3. Text Preprocessing
4. TF-IDF Vectorization
5. Model Training
6. Model Evaluation
7. Save Model using Pickle
8. Integrate Model with Django
9. Deploy Web Application

---

## 📸 Screenshots

Add screenshots of:

* Home Page
* Prediction Page
* Prediction Result
* Admin Panel (Optional)

Create a folder named **screenshots** and place your images there.

---

## 📈 Future Improvements

* User Authentication
* News API Integration
* Confidence Score
* Explainable AI Predictions
* Model Retraining Pipeline
* Docker Support
* Cloud Deployment

---

## 🤝 Contributing

Contributions are welcome. Feel free to fork this repository and submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Tanish Tushid**

* Python Developer
* Machine Learning Enthusiast
* Django Developer

If you found this project useful, don't forget to ⭐ star the repository!
