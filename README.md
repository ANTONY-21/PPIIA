# 🏛️ Public Policy Insight & Impact Analyzer (PPIIA)

**Simplifying government Bills for every citizen — using 100% local AI.**

Upload a Bill (PDF, URL, or raw text) and get a structured, plain-English analysis covering:

- Citizen-friendly summary (8th-grade reading level)
- Objective & owning ministry
- Affected sectors / industries
- Chronology — predecessor acts & amendments
- Short / medium / long-term impact
- Positives, negatives, risks

**No API keys. No cloud. No paid services.** Runs entirely on your CPU using
open-source Hugging Face models.

---

## 1. Problem statement
Government Bills are written in dense legal language. Most citizens don't know what a new law actually *does*, which sectors it affects, or what its short- and long-term consequences may be. PPIIA turns that dense text into an easy-to-read, structured report anyone can understand.

## 2. Approach
| Step | Technique |
|---|---|
| Ingestion | `pdfplumber` for PDFs, `requests + BeautifulSoup` for URLs / HTML |
| Cleaning | regex-based whitespace + page-number removal |
| Chunking | word-based splitter for map-reduce summarization |
| Entity extraction | regex for years / acts / ministries |
| Summarization | `sshleifer/distilbart-cnn-12-6` (local, CPU) |
| Sector classification | `valhalla/distilbart-mnli-12-3` zero-shot (local, CPU) |
| Impact splitting | heuristic thirds over the generated summary |

## 3. Local AI — how it works

- **First run** downloads ~700 MB of Hugging Face models into your local cache
  (`~/.cache/huggingface/`)
- **Every subsequent run** is fully offline — no network calls, no API keys
- Uses **map-reduce summarization** so it can handle Bills far longer than
  BART's 1024-token context window
- Uses **zero-shot classification** against 15 candidate sectors, so a Bill
  can be tagged with multiple sectors without any fine-tuning

## 4. Project structure
```
PPIIA/
├── ppiia.py                 # core analysis pipeline (modular functions)
├── app.py                   # Streamlit dashboard
├── ppiia_colab.ipynb        # Colab notebook + Gradio demo
├── requirements.txt
├── .gitignore
├── README.md
└── INTERVIEW_GUIDE.md       # live-evaluation study guide
```

## 5. How to run

### Option A — Google Colab (zero setup, recommended)
1. Open https://colab.research.google.com → File → Upload notebook → `ppiia_colab.ipynb`
2. Run Cell 1 (installs dependencies)
3. Run Cell 2 and upload `ppiia.py` + your sample Bill PDF
4. Run Cells 3–4 to warm up the models and run the analyzer
5. Run Cell 5 to launch the **Gradio public URL** for your demo video

### Option B — Local Streamlit
```bash
pip install -r requirements.txt
streamlit run app.py
```
First run downloads ~700 MB of models into the HF cache. After that everything
runs offline.

### Option C — CLI (batch JSON output)
```bash
python ppiia.py path/to/bill.pdf --out outputs/report.json
python ppiia.py https://sansad.in/ls/.../bill.pdf
```

## 6. Example output
```json
{
  "title": "The Digital Personal Data Protection Bill, 2023",
  "ministry": "Ministry of Electronics and Information Technology",
  "objective": "To regulate the processing of digital personal data...",
  "summary": "This Bill sets rules for how companies and the government...",
  "sectors": ["Technology", "Finance", "Healthcare"],
  "chronology": ["IT Act, 2000", "Aadhaar Act, 2016"],
  "impacts": {
    "short_term": "New compliance officer roles in every big company.",
    "medium_term": "Stricter data breach penalties.",
    "long_term": "Citizens gain real ownership of their digital data."
  }
}
```

## 7. Business use cases
| User | How PPIIA helps |
|---|---|
| Citizens | Understand new laws without legal training |
| Journalists | Get instant structured bill briefings |
| NGOs | Spot clauses that affect vulnerable groups |
| Startups / corporates | Know which compliance rules apply to them |
| Educators | Teach civics with real, digestible examples |

## 8. Tech tags
`Python` `Transformers` `HuggingFace` `Local AI` `CPU` `NLP` `Summarization` `Zero-Shot Classification` `Legal NLP` `Streamlit` `Gradio`
