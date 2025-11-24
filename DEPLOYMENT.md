# Quick Deployment Guide

## ⚠️ Why Vercel Gives 404 Error

**Vercel is NOT suitable for this application** because:
1. **10-second timeout** - Your app needs 7-10s to initialize models
2. **Serverless limitations** - Background threading doesn't work
3. **Cold starts** - Every request after inactivity will timeout
4. **Large dependencies** - sentence-transformers is too big

## ✅ Recommended: Deploy to Render (FREE)

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Add deployment configs"
git push
```

### Step 2: Deploy on Render
1. Go to [render.com](https://render.com)
2. Sign up/Login with GitHub
3. Click "New +" → "Web Service"
4. Connect your `Medical_Chatbot` repository
5. Render will auto-detect `render.yaml`

### Step 3: Add Environment Variables
In Render dashboard, add:
- `PINECONE_API_KEY` = your_key
- `GROQ_API_KEY` = your_key
- `PINECONE_INDEX_NAME` = medical-chatbot

### Step 4: Deploy!
Click "Create Web Service" - Done! 🎉

**Your app will be live at**: `https://medical-chatbot-xxxx.onrender.com`

## Alternative: Railway

1. Go to [railway.app](https://railway.app)
2. "New Project" → "Deploy from GitHub"
3. Select repository
4. Add environment variables
5. Deploy

## Files Created for You

✅ `vercel.json` - Vercel config (not recommended)
✅ `render.yaml` - Render config (RECOMMENDED)
✅ `requirements.txt` - Updated with gunicorn
✅ `.vercelignore` - Exclude files
✅ `.gitignore` - Updated

## What to Do Now

1. **Commit the new files**:
   ```bash
   git add vercel.json render.yaml requirements.txt .vercelignore .gitignore
   git commit -m "Add deployment configurations"
   git push
   ```

2. **Choose a platform**:
   - **Render** (recommended) - Free, easy, no timeout issues
   - **Railway** - Also good, free tier
   - **Vercel** - Will have issues, not recommended

3. **Deploy and test**!

## Need Help?

Check the detailed deployment guide in the artifacts for troubleshooting!
