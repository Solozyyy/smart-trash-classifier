# 🚀 Streamlit Deployment Guide - Chi Tiết Từng Bước

## ✅ ĐÃ HOÀN THÀNH

Tôi đã tạo sẵn cho bạn:
- ✅ `app/streamlit_app.py` - Full Streamlit app (200+ lines)
- ✅ `app/requirements.txt` - Dependencies
- ✅ `app/README.md` - Documentation

## 📝 CÁC BƯỚC BẠN CẦN LÀM

### **BƯỚC 1: Test Local (10 phút)**

#### 1.1. Mở Terminal tại thư mục project
```bash
cd d:\hoc_AI\smart-trash-classifier
```

#### 1.2. Install Streamlit (lần đầu tiên)
```bash
pip install streamlit
```

#### 1.3. Run app
```bash
cd app
streamlit run streamlit_app.py
```

#### 1.4. Test app
- Browser sẽ tự động mở tại `http://localhost:8501`
- Upload 1 ảnh test (có thể từ data/)
- Check xem prediction có chạy không

**⚠️ LƯU Ý:**
- Lần đầu chạy sẽ hơi chậm (loading model)
- Nếu lỗi "model not found", check xem file `weights/best_model.h5` có tồn tại không

---

### **BƯỚC 2: Push lên GitHub (5 phút)**


#### 2.1. Check Git status
```bash
git status
```

#### 2.2. Add files
```bash
git add app/
git add STREAMLIT_DEPLOYMENT_GUIDE.md
```

#### 2.3. Commit
```bash
git commit -m "Add Streamlit web app for waste classification"
```

#### 2.4. Push
```bash
git push origin main
```

**⚠️ QUAN TRỌNG:** 
- File `best_model.h5` (~98MB) quá lớn để push lên GitHub
- Bạn cần dùng **Git LFS** hoặc **Google Drive link**

---

### **BƯỚC 3: Xử Lý Model File (15 phút)**

#### Option A: Git LFS (Recommended)

**3.1. Install Git LFS:**
```bash
# Download từ: https://git-lfs.github.com/
# Hoặc:
git lfs install
```

**3.2. Track model file:**
```bash
git lfs track "*.h5"
git add .gitattributes
```

**3.3. Add và push model:**
```bash
git add weights/best_model.h5
git commit -m "Add trained model with Git LFS"
git push
```

#### Option B: Google Drive Link (Easier)

**3.1. Upload model lên Google Drive:**
- Upload `weights/best_model.h5` lên Drive
- Set permission: "Anyone with link can view"
- Copy link

**3.2. Tạo script download trong app:**

Tạo file `app/download_model.py`:
```python
import gdown
import os

def download_model():
    url = 'YOUR_GOOGLE_DRIVE_LINK'  # Replace with your link
    output = '../weights/best_model.h5'
    
    if not os.path.exists(output):
        print("Downloading model...")
        gdown.download(url, output, quiet=False)
        print("Model downloaded!")
    else:
        print("Model already exists")

if __name__ == "__main__":
    download_model()
```

**3.3. Update requirements.txt:**
```bash
echo "gdown==4.7.1" >> app/requirements.txt
```

**3.4. Update streamlit_app.py:**
```python
# Add at the top after imports
from download_model import download_model
download_model()  # Download model if not exists
```

---

### **BƯỚC 4: Deploy lên Streamlit Cloud (10 phút)**

#### 4.1. Truy cập Streamlit Cloud
- Go to: https://share.streamlit.io/
- Click "Sign in with GitHub"
- Authorize Streamlit

#### 4.2. Create New App
- Click "New app" button
- Fill in:
  - **Repository:** `Solozyyy/smart-trash-classifier` (your repo)
  - **Branch:** `main`
  - **Main file path:** `app/streamlit_app.py`

#### 4.3. Advanced Settings (Optional)
- Click "Advanced settings"
- Python version: 3.9
- Can add secrets if needed

#### 4.4. Deploy!
- Click "Deploy!"
- Wait 2-5 minutes for deployment
- You'll get a URL like: `https://your-app-name.streamlit.app`

#### 4.5. Test Deployed App
- Open the URL
- Upload test image
- Check if everything works

---

### **BƯỚC 5: Update README với Demo Link (5 phút)**


Edit `README.md` and add at the top:

```markdown
# Smart Trash Classifier

![Status](https://img.shields.io/badge/Status-Live-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-93.63%25-blue)

> AI-powered waste classification with 93.63% accuracy

## 🌐 Live Demo

**Try it now:** [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)

Upload an image and get instant classification with recycling instructions!

---
```

Commit và push:
```bash
git add README.md
git commit -m "Add live demo link"
git push
```

---

## 🎯 CHECKLIST HOÀN CHỈNH

### ✅ Local Development
- [ ] Install Streamlit: `pip install streamlit`
- [ ] Run app: `streamlit run app/streamlit_app.py`
- [ ] Test with sample images
- [ ] Verify predictions work
- [ ] Check UI rendering

### ✅ GitHub Push
- [ ] Add files: `git add app/`
- [ ] Commit: `git commit -m "Add Streamlit app"`
- [ ] Push: `git push`
- [ ] Verify files on GitHub

### ✅ Model Handling
- [ ] Choose method: Git LFS or Google Drive
- [ ] Upload/track model file
- [ ] Test model loading in app

### ✅ Streamlit Cloud Deployment
- [ ] Sign in: https://share.streamlit.io/
- [ ] Create new app
- [ ] Configure: repo, branch, main file
- [ ] Deploy and wait
- [ ] Test deployed app

### ✅ Documentation Update
- [ ] Add demo link to README
- [ ] Update badges
- [ ] Push changes
- [ ] Verify link works

---

## 🐛 TROUBLESHOOTING

### Problem 1: Model file too large for GitHub
**Solution:** Use Git LFS or Google Drive link (see Bước 3)

### Problem 2: App crashes on Streamlit Cloud
**Check logs:**
- Click "Manage app" on Streamlit Cloud
- View logs
- Common issues:
  - Model file not found → Use Google Drive download
  - Memory limit → Model too large for free tier

### Problem 3: Slow first prediction
**This is normal:**
- First prediction loads model (~5-10 seconds)
- Subsequent predictions are fast (~45ms)
- Use `@st.cache_resource` (already in code)

### Problem 4: Import errors
**Solution:**
```bash
# Make sure all dependencies in requirements.txt
pip install -r app/requirements.txt
```

---

## 📊 AFTER DEPLOYMENT

### Update CV
```
✅ Deployed interactive web application using Streamlit with 
   drag-and-drop upload, real-time prediction, and recycling 
   guidance
   
📱 Live Demo: https://your-app.streamlit.app
```

### Share on LinkedIn
```
🎉 Excited to share my latest project!

Built an AI-powered waste classifier that:
✅ Identifies 12 types of waste with 93.63% accuracy
✅ Provides recycling instructions
✅ Shows environmental impact

Try it here: [your link]

Tech: Python, TensorFlow, ResNet50, Streamlit

#AI #MachineLearning #DeepLearning #Environment
```

---

## ⏱️ TIME ESTIMATE

- **Local test:** 10 minutes
- **Git push:** 5 minutes
- **Model handling:** 15 minutes
- **Streamlit Cloud:** 10 minutes
- **Update docs:** 5 minutes

**Total: ~45 minutes to 1 hour**

---

## 🎯 NEXT STEPS

After successful deployment:

1. **Test thoroughly** with various images
2. **Share link** with friends for feedback
3. **Monitor** Streamlit Cloud analytics
4. **Update CV** with demo link
5. **Prepare** for Phase 2: FastAPI or Grad-CAM

---

## 💡 PRO TIPS

1. **Use Git LFS** for large files (best practice)
2. **Test locally first** before deploying
3. **Check logs** if deployment fails
4. **Free tier limits:** 1GB RAM, might need to optimize model
5. **Custom domain:** Available on Streamlit paid plans

---

**🎉 Once deployed, you'll have a public URL to show recruiters!**

Good luck! 🚀
