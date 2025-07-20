# 💙 EmpathyAI 2.0 - Professional Mental Health Assistant

A modular, production-ready AI system for empathetic conversation and emotional support. Built with advanced NLP, multi-model emotion detection, and secure user authentication.

![EmpathyAI Architecture](https://img.shields.io/badge/Architecture-Modular-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen) ![Streamlit](https://img.shields.io/badge/Streamlit-1.42%2B-red) ![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- Google Cloud account (for Gemini API)
- Optional: n8n instance for analytics
- Optional: Google Sheets for cloud storage

## 📁 Project Structure

```
empathy_ai/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── test_system.py             # Comprehensive test suite
├── .gitignore                 # Git ignore rules
├── .streamlit/
│   └── secrets.toml           # Configuration secrets (DO NOT COMMIT)
└── src/
    ├── __init__.py            # Package initialization
    ├── emotion.py             # Emotion detection using HuggingFace
    ├── sentiment_fusion.py    # Multi-model sentiment analysis
    ├── llm_response.py        # Google Gemini API integration
    ├── response_generator.py  # Empathetic response generation
    ├── memory.py              # SQLite/Google Sheets memory management
    ├── n8n_integration.py     # Webhook integration for analytics
    └── auth.py                # Google OAuth & simple authentication
```

## 🔧 Step-by-Step Integration Guide

### Step 1: Environment Setup

1. **Clone or create your project directory:**
   ```bash
   mkdir empathy_ai
   cd empathy_ai
   ```

2. **Create a Python virtual environment:**
   ```bash
   python -m venv venv

   # On Windows:
   venv\Scripts\activate

   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Step 2: API Configuration

#### Google Gemini API Setup (Required)

1. **Get Gemini API Key:**
   - Go to [Google AI Studio](https://aistudio.google.com/)
   - Click "Get API Key" → "Create API Key"
   - Copy your API key

2. **Configure the API key:**
   - Open `.streamlit/secrets.toml`
   - Add your API key:
     ```toml
     GEMINI_API_KEY = "your_actual_api_key_here"
     ```

#### Google OAuth Setup (Optional - for production auth)

1. **Create Google Cloud Project:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing

2. **Enable Google+ API:**
   - Go to "APIs & Services" → "Library"
   - Search for "Google+ API" and enable it

3. **Create OAuth Credentials:**
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "OAuth client ID"
   - Choose "Web application"
   - Add redirect URI: `http://localhost:8501`

4. **Download credentials:**
   - Download the JSON file as `google_credentials.json`
   - Or add to `secrets.toml`:
     ```toml
     google_client_id = "your_client_id.apps.googleusercontent.com"
     google_client_secret = "your_client_secret"
     cookie_key = "your_random_secret_key_here"
     ```

### Step 3: Test the System

1. **Run the comprehensive test suite:**
   ```bash
   python test_system.py
   ```

2. **Expected output:**
   ```
   🔍 EmpathyAI 2.0 - System Test Suite
   ==================================================

   📦 Testing Module Imports...
     ✅ All modules imported successfully

   🎭 Testing Emotion Detection...
     📝 'I am so happy today!' -> joy (0.89)
     ✅ Emotion detection working

   🔀 Testing Sentiment Fusion...
     ✅ Sentiment fusion working

   🤖 Testing LLM Response...
     ✅ LLM response generation working

   💬 Testing Response Generator...
     ✅ Response generator working

   💾 Testing Memory System...
     ✅ Memory system working

   🔗 Testing n8n Integration...
     ⚠️  n8n not configured (expected for development)

   🔐 Testing Auth System...
     ✅ Auth system initialized

   📊 Overall: 8/8 tests passed
   🎉 All tests passed! EmpathyAI is ready for deployment.
   ```

### Step 4: Run the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   - Navigate to `http://localhost:8501`
   - You should see the EmpathyAI interface

3. **Test the chat functionality:**
   - Enter a username (simple auth)
   - Start chatting with the AI
   - Observe emotion detection and responses

## 🔧 Advanced Configuration

### n8n Analytics Integration (Optional)

1. **Set up n8n:**
   - Install n8n: `npm install n8n -g`
   - Start n8n: `n8n start`

2. **Create webhook workflow:**
   - Create a new workflow
   - Add "Webhook" trigger node
   - Copy the webhook URL

3. **Configure in EmpathyAI:**
   ```toml
   N8N_WEBHOOK_URL = "http://localhost:5678/webhook/empathy-ai"
   ```

### Google Sheets Integration (Optional)

1. **Create service account:**
   - Go to Google Cloud Console
   - "IAM & Admin" → "Service Accounts"
   - Create new service account

2. **Download credentials:**
   - Generate JSON key file
   - Save as `gcp_service_account.json`

3. **Create Google Sheet:**
   - Create new Google Sheet named "EmpathyAI_Memory"
   - Share with service account email

4. **Configure in secrets:**
   ```toml
   USE_SHEETS = true
   SHEET_NAME = "EmpathyAI_Memory"
   ```

## 🚀 Deployment to Streamlit Cloud

### Step 1: Prepare Repository

1. **Create GitHub repository:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: EmpathyAI 2.0"
   git remote add origin https://github.com/yourusername/empathy-ai.git
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to [Streamlit Cloud](https://share.streamlit.io/)**
2. **Connect your GitHub repository**
3. **Configure deployment:**
   - App file: `app.py`
   - Python version: 3.8+

### Step 3: Configure Secrets in Cloud

1. **Go to App Settings → Advanced Settings**
2. **Add all secrets from your local `secrets.toml`:**
   ```toml
   GEMINI_API_KEY = "your_actual_api_key"
   google_client_id = "your_client_id"
   google_client_secret = "your_client_secret"
   cookie_key = "your_random_secret"
   redirect_uri = "https://your-app-name.streamlit.app"
   ```

## 🧪 Testing Each Component

### Test Emotion Detection
```python
from src.emotion import detect_emotion

result = detect_emotion("I'm feeling really happy today!")
print(result)  # {'label': 'joy', 'confidence': 0.89, ...}
```

### Test Sentiment Fusion
```python
from src.sentiment_fusion import fuse_sentiment_emotion

fused = fuse_sentiment_emotion("I love this!", "joy")
print(fused)  # "positive-joy"
```

### Test LLM Response
```python
from src.llm_response import ask_gemini

response = ask_gemini("How can I help someone who is sad?")
print(response)  # Empathetic AI response
```

### Test Memory System
```python
from src.memory import create_memory_manager

memory = create_memory_manager("test_user")
memory.add_emotion_record("joy", 0.8, "I'm happy!")
records = memory.get_recent_emotions()
print(records)
```

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found" errors:**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Gemini API not working:**
   - Check API key in secrets.toml
   - Verify Google AI Studio quota
   - Try: `python -c "from src.llm_response import check_api_health; print(check_api_health())"`

3. **Google Auth issues:**
   - Verify OAuth credentials
   - Check redirect URI matches exactly
   - Try simple auth fallback first

4. **Memory/Database errors:**
   - Check file permissions
   - Delete `empathy_memory.db` and restart
   - Try Google Sheets alternative

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Considerations

- **Model Loading:** First run takes longer due to HuggingFace model downloads
- **Memory Usage:** ~2-4GB for all models in memory
- **Response Time:** 2-5 seconds per message with Gemini API
- **Database:** SQLite handles 1000s of records efficiently

## 🔐 Security Best Practices

1. **Never commit secrets:**
   - Use `.gitignore` for `secrets.toml`
   - Use environment variables in production

2. **API Key Security:**
   - Rotate Gemini API keys regularly
   - Use restricted scopes where possible

3. **User Data:**
   - SQLite database is local by default
   - Google Sheets requires proper IAM permissions
   - Consider data encryption for sensitive deployments

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `python test_system.py`
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push to branch: `git push origin feature/amazing-feature`
6. Open pull request

## 📈 Roadmap

- [ ] Voice input/output integration
- [ ] Mood tracking visualizations
- [ ] Integration with mental health APIs
- [ ] Multi-language support
- [ ] Advanced conversation analytics
- [ ] Mobile app companion

## 📄 License

MIT License - see LICENSE file for details.

## 🙋‍♂️ Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/empathy-ai/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/empathy-ai/discussions)
- **Email:** your.email@example.com

---

**Built with ❤️ for AI4Good - Making mental health support accessible to everyone.**
