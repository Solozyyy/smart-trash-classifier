# 🗑️ Smart Trash Classifier - Streamlit Web App

Interactive web application for waste classification with AI.

## 🚀 Quick Start

### Local Development

1. **Install dependencies:**
```bash
cd app
pip install -r requirements.txt
```

2. **Run the app:**
```bash
streamlit run streamlit_app.py
```

3. **Open browser:**
- App will open automatically at `http://localhost:8501`

## 📦 What's Inside

- `streamlit_app.py` - Main application with UI and prediction logic
- `requirements.txt` - Python dependencies
- `README.md` - This file

## 🌐 Deploy to Streamlit Cloud (FREE)

1. **Push code to GitHub:**
```bash
git add app/
git commit -m "Add Streamlit web app"
git push
```

2. **Go to Streamlit Cloud:**
- Visit https://streamlit.io/cloud
- Sign in with GitHub
- Click "New app"

3. **Configure deployment:**
- Repository: `your-username/smart-trash-classifier`
- Branch: `main`
- Main file path: `app/streamlit_app.py`

4. **Deploy:**
- Click "Deploy!"
- Wait 2-3 minutes
- Get your public URL!

## ✨ Features

- ✅ Drag & drop image upload
- ✅ Real-time prediction with confidence scores
- ✅ Top 3 predictions visualization
- ✅ Recycling instructions for each category
- ✅ Environmental impact information
- ✅ Inference time tracking
- ✅ Responsive design

## 📊 Model Info

- **Architecture:** ResNet50 with Transfer Learning
- **Accuracy:** 93.63%
- **Training Time:** 58 minutes
- **Dataset:** 24,000 balanced images
- **Categories:** 12 waste types

## 🎯 Supported Categories

1. Battery
2. Biological (Organic)
3. Brown Glass
4. Cardboard
5. Clothes
6. Green Glass
7. Metal
8. Paper
9. Plastic
10. Shoes
11. Trash (General)
12. White Glass

## 🐛 Troubleshooting

**Issue: Model not loading**
- Ensure `best_model.h5` is in `weights/` folder
- Check path in `streamlit_app.py` line 156

**Issue: Dependencies error**
- Run `pip install -r requirements.txt` again
- Use Python 3.8+

**Issue: App runs slowly**
- First prediction is slow (model loading)
- Subsequent predictions are fast (~45ms)

## 📝 License

MIT License
