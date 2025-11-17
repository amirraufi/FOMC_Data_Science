# Fine-Tuning AI Models & Creating Interactive Frontend

Complete guide for advanced improvements to your FOMC analysis project.

---

## Part 1: Fine-Tuning AI Models

### Option A: Fine-Tune FinBERT on FOMC Data

**Why**: FinBERT is pre-trained on general financial text. Fine-tuning on FOMC-specific language will improve hawkishness detection.

**Steps**:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import pandas as pd

# 1. Prepare labeled data
df = pd.read_csv('communications.csv')

# Create labels from market reactions (1 = hawkish/yields up, 0 = dovish/yields down)
df['label'] = (df['dy2_1d_bp'] > 0).astype(int)

# Split text into chunks (FinBERT has 512 token limit)
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')

# 2. Create dataset
from torch.utils.data import Dataset

class FOMCDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 3. Fine-tune
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=2)

training_args = TrainingArguments(
    output_dir='./fomc_finbert_finetuned',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
trainer.save_model('./fomc_finbert_finetuned')
```

**Expected Improvement**: 5-10% better directional accuracy

---

### Option B: Fine-Tune GPT-3.5/GPT-4 (via OpenAI API)

**Why**: Get more FOMC-specific hawkishness scores

**Steps**:

```python
import openai

# 1. Prepare training data (JSONL format)
training_data = []
for idx, row in df.iterrows():
    training_data.append({
        "messages": [
            {"role": "system", "content": "You are a Federal Reserve expert analyzing FOMC statements for hawkishness."},
            {"role": "user", "content": f"Rate this statement's hawkishness (1-5): {row['Text']}"},
            {"role": "assistant", "content": f"{row['actual_hawkishness_score']}"}  # You need ground truth
        ]
    })

# Save to JSONL
import json
with open('fomc_training.jsonl', 'w') as f:
    for item in training_data:
        f.write(json.dumps(item) + '\n')

# 2. Upload and fine-tune via OpenAI API
file = openai.File.create(
    file=open("fomc_training.jsonl", "rb"),
    purpose='fine-tune'
)

fine_tune = openai.FineTune.create(
    training_file=file['id'],
    model="gpt-3.5-turbo"
)

# 3. Use fine-tuned model
response = openai.ChatCompletion.create(
    model=fine_tune['fine_tuned_model'],
    messages=[
        {"role": "user", "content": "Rate hawkishness: The Committee decided to raise rates..."}
    ]
)
```

**Cost**: ~$0.10-1.00 per 1K examples
**Expected Improvement**: More consistent, FOMC-specific scores

---

### Option C: Train Custom Transformer (Advanced)

**For maximum performance**:

```python
from transformers import BertForSequenceClassification, BertTokenizer, Trainer
import torch

# Architecture: BERT encoder → Regression head → Yield prediction
class FOMCYieldPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.regressor = nn.Linear(768, 1)  # Predict yield change directly

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        yield_pred = self.regressor(dropout_output)
        return yield_pred

# Train end-to-end on FOMC text → Yield changes
```

**Training Time**: 2-4 hours on GPU
**Expected Improvement**: 10-15% better than feature-based models

---

## Part 2: Creating Interactive Frontend

### Option A: Streamlit (Easiest) ⭐ **Recommended**

Already created for you: `app_streamlit.py`

**Run it**:
```bash
pip install streamlit plotly
streamlit run app_streamlit.py
```

**Features**:
- ✅ Paste FOMC statement → Get prediction
- ✅ Interactive charts
- ✅ Feature importance visualization
- ✅ Historical comparison
- ✅ Works on any device (desktop/mobile)

**Deploy to web** (free):
```bash
# 1. Push to GitHub
git push

# 2. Go to streamlit.io/cloud
# 3. Connect GitHub repo
# 4. Deploy with one click
# 5. Get public URL: https://your-app.streamlit.app
```

---

### Option B: Gradio (ML-Focused)

```python
import gradio as gr

def predict_reaction(statement_text, model_choice):
    # Your prediction logic
    prediction = model.predict([statement_text])[0]
    confidence = 0.85

    return {
        "Predicted Change": f"{prediction:.2f} bp",
        "Confidence": f"{confidence:.1%}",
        "Direction": "Hawkish" if prediction > 0 else "Dovish"
    }

# Create interface
demo = gr.Interface(
    fn=predict_reaction,
    inputs=[
        gr.Textbox(label="FOMC Statement", lines=10),
        gr.Dropdown(["Random Forest", "XGBoost", "Neural Net"], label="Model")
    ],
    outputs=[
        gr.JSON(label="Prediction")
    ],
    title="FOMC Market Reaction Predictor",
    description="Predict Treasury yield reactions using AI",
    examples=[
        ["The Committee decided to raise the target range...", "Random Forest"],
        ["Inflation has eased and the Committee decided to lower...", "XGBoost"]
    ]
)

demo.launch(share=True)  # Creates public URL
```

**Deploy**: `gradio app.py` → Get shareable link

---

### Option C: Full Web App (React + FastAPI)

**Backend (FastAPI)**:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

class Statement(BaseModel):
    text: str
    model_name: str = "RandomForest"

@app.post("/predict")
def predict(statement: Statement):
    # Extract features
    features = extract_features(statement.text)

    # Predict
    prediction = model.predict([features])[0]

    return {
        "predicted_change": float(prediction),
        "confidence": 0.85,
        "direction": "Hawkish" if prediction > 0 else "Dovish"
    }

# Run: uvicorn app:app --reload
```

**Frontend (React)**:
```javascript
function PredictForm() {
  const [statement, setStatement] = useState('');
  const [prediction, setPrediction] = useState(null);

  const handleSubmit = async () => {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: statement })
    });

    const data = await response.json();
    setPrediction(data);
  };

  return (
    <div>
      <textarea value={statement} onChange={(e) => setStatement(e.target.value)} />
      <button onClick={handleSubmit}>Predict</button>
      {prediction && (
        <div>
          <h3>Prediction: {prediction.predicted_change} bp</h3>
          <p>Direction: {prediction.direction}</p>
        </div>
      )}
    </div>
  );
}
```

**Deployment**: Vercel (frontend) + Railway/Heroku (backend)

---

## Part 3: Advanced Features for Frontend

### 1. Real-Time FOMC Feed
```python
import feedparser

# Monitor Fed website RSS
feed = feedparser.parse('https://www.federalreserve.gov/feeds/press_all.xml')

for entry in feed.entries:
    if 'FOMC' in entry.title:
        # Auto-analyze new statements
        prediction = predict(entry.summary)
        send_alert(prediction)
```

### 2. Historical Backtesting
```python
def backtest_strategy(df, model):
    """
    Simulate trading based on predictions
    """
    df['position'] = np.sign(df['predicted_change'])  # Buy if yields up, sell if down
    df['pnl'] = df['position'] * df['actual_change']
    df['cumulative_pnl'] = df['pnl'].cumsum()

    return {
        'total_return': df['pnl'].sum(),
        'sharpe_ratio': df['pnl'].mean() / df['pnl'].std(),
        'win_rate': (df['pnl'] > 0).mean()
    }
```

### 3. Uncertainty Quantification
```python
from sklearn.ensemble import GradientBoostingRegressor

# Quantile regression for prediction intervals
models = {
    'lower': GradientBoostingRegressor(loss='quantile', alpha=0.1),
    'median': GradientBoostingRegressor(loss='quantile', alpha=0.5),
    'upper': GradientBoostingRegressor(loss='quantile', alpha=0.9)
}

# Predict with 80% confidence interval
prediction = {
    'point': models['median'].predict(X)[0],
    'lower_bound': models['lower'].predict(X)[0],
    'upper_bound': models['upper'].predict(X)[0]
}
```

---

## Part 4: Recommended Implementation Path

### Phase 1: Quick Win (1 day)
1. ✅ Run `create_plots.py` → Generate all figures
2. ✅ Run `streamlit run app_streamlit.py` → Interactive demo
3. ✅ Deploy to Streamlit Cloud → Public URL

### Phase 2: Improve Models (1 week)
1. Run `test_different_horizons.py` → Find optimal prediction window
2. Fine-tune FinBERT on FOMC data
3. Add XGBoost/LightGBM models
4. Implement uncertainty quantification

### Phase 3: Advanced Frontend (1-2 weeks)
1. Add real-time FOMC feed monitoring
2. Build historical backtesting tool
3. Create interactive case studies
4. Add export to PDF report feature

### Phase 4: Paper Integration (ongoing)
1. Use figures from `create_plots.py` in paper
2. Reference Streamlit app in paper (live demo)
3. Show backtesting results for economic significance
4. Include uncertainty intervals in robustness checks

---

## Resources

**Streamlit**:
- Docs: https://docs.streamlit.io
- Gallery: https://streamlit.io/gallery
- Deploy: https://streamlit.io/cloud

**Fine-Tuning**:
- Hugging Face: https://huggingface.co/docs/transformers/training
- OpenAI: https://platform.openai.com/docs/guides/fine-tuning

**Deployment**:
- Streamlit Cloud (free): https://streamlit.io/cloud
- Heroku (backend): https://www.heroku.com
- Vercel (frontend): https://vercel.com

---

## Next Steps

1. **Try Streamlit app**: `streamlit run app_streamlit.py`
2. **Test different horizons**: `python test_different_horizons.py`
3. **Generate all plots**: `python create_plots.py`
4. **Deploy**: Push to Streamlit Cloud for public demo
5. **Fine-tune**: Start with FinBERT fine-tuning

**Timeline**: You can have a working demo online in < 1 hour using Streamlit!
