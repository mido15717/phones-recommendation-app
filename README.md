# Phone-Recommendation-App

This is a simple phone recommendation app that helps users find the best phone based on their preferences and budget.

The system uses a Retrieval-Augmented Generation (RAG) pipeline: it understands what you're looking for (budget, brand, use case, etc.), searches a local database of real phone listings, and recommends grounded, relevant matches through a conversational chat interface — no made-up specs, only real products from the catalog.

## Requirements

- python 3.11.15

### Installation

1. Clone the repository:
```bash
   git clone https://github.com/yourusername/phone-recommendation-app.git
   cd phone-recommendation-app
```

2. download and install Miniconda or Anaconda from the official website: https://www.anaconda.com/docs/getting-started/miniconda/install
```bash
   conda create -n phone-recommendation-app python=3.11
   conda activate phone-recommendation-app
```

3. Install the required dependencies:
```bash
   pip install -r requirements.txt
```

4. Setup environment variables
```bash
   $ cp .env .env.example
```
Set your environment variables in the .env like your API_KEY value.

5. Run FastAPI server
```bash
   $ uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

6. Run the UI (in a separate terminal)
```bash
   $ streamlit run src/ui/app.py
```

7. Open the app
   - UI: http://localhost:8501
   - API docs: http://localhost:8000/docs

## How it works

1. **Understand your request** — the system asks about your budget, preferred brand, and what you'll use the phone for.
2. **Search the catalog** — your preferences are matched against a database of real phone listings using vector search (RAG).
3. **Get grounded recommendations** — the assistant replies with actual phones from the catalog that fit what you asked for, along with why they're a good match.

## Project Structure

```
src/
├── routes/       # API endpoints
├── services/      # Core recommendation, RAG, and data logic
├── ui/           # Streamlit chat interface
└── main.py       # FastAPI entry point
```