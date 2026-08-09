# 🎤 Live Evaluation Study Guide — PPIIA

Your script for the 30-min live evaluation: **3 min intro + 12 min walkthrough + 10 min mock Q&A + 5 min feedback.** Read it 2–3 times and you'll be able to defend every line of code.

---

## 0. Self introduction (3 mins)

> "Hi, I'm **[name]**. I'm pursuing the AIML track at Guvi. I'm comfortable with Python, NLP, and the Hugging Face ecosystem, and I'm now exploring **LLM-powered applications**. My project is **PPIIA — the Public Policy Insight & Impact Analyzer**. It takes any government Bill PDF or URL and produces a structured, citizen-friendly analysis using a hybrid pipeline of Hugging Face transformers and a large language model. I built a Streamlit dashboard and a Gradio demo around it."

---

## 1. Project walkthrough script (12 mins)

Follow the **Capstone Explanation Guideline** order: domain → problem → objective → ingestion → preprocessing → models → outputs → deployment → conclusion → business use.

### 1.1 Domain (3 lines)
> "The domain is **CivicTech and Public Policy Analytics**. Indian Parliament passes hundreds of bills written in dense legal language that ordinary citizens can't follow. NLP and large language models can bridge that gap by turning legal text into plain-English summaries that anyone can read."

### 1.2 Project introduction (2 lines)
> "PPIIA is an AI platform that ingests a Bill PDF or URL, extracts and cleans the text, and produces a structured JSON report covering objective, summary, affected sectors, chronology, impacts, and risks — all in 8th-grade English."

### 1.3 Objective (2 lines)
> "To build a reproducible, modular pipeline that converts any Indian government Bill into a citizen-friendly briefing, with two operating modes: a 100% free local mode using Hugging Face transformers, and an upgraded LLM mode using the Anthropic Claude API with structured outputs."

### 1.4 Problem statement (1 line)
> "Citizens, journalists, and small businesses can't keep up with new legislation because the Bills are too long and too legalistic — PPIIA solves that with NLP + LLM summarization."

### 1.5 Data ingestion
- **PDF input** — `pdfplumber` (preferred) with a `pypdf` fallback. `pdfplumber` handles columns and tables better than the older PyPDF2.
- **URL input** — `requests` for fetching. If the response Content-Type is `application/pdf`, write to a temp file and reuse the PDF extractor; otherwise parse HTML with `BeautifulSoup`, stripping `script`, `style`, `nav`, `footer`, `header` tags before pulling text.
- **Raw text input** — passed straight through to the cleaner.

### 1.6 Cleaning + chunking
- `clean_text()` collapses whitespace with `re.sub(r"\s+", " ", text)`, removes "Page X of Y" headers, and strips standalone page numbers
- `chunk_text()` is a **word-based** splitter — keeps paragraph boundaries, makes chunks of ~900 words. We need chunks because BART has a 1024-token context window
- I deliberately did NOT use NLTK/spaCy for tokenization to keep dependencies minimal

### 1.7 Entity extraction (regex)
- **Years** — `\b(19|20)\d{2}\b`
- **Acts** — `\b([A-Z][A-Za-z,\s]+?(?:Act|Bill|Ordinance)(?:\s*,?\s*\d{4})?)\b`
- **Ministries** — `Ministry of [A-Z][A-Za-z &]+?`
- These run in **both** modes because they're free, fast, and don't need a model

### 1.8 The two operating modes (the key design decision)

| | FREE (local) mode — `ppiia.py` | Claude AI mode — `ppiia_ai.py` |
|---|---|---|
| Trigger | Engine picker in the sidebar; no key needed | `ANTHROPIC_API_KEY` in `.env` or pasted in the sidebar |
| Summarization | `sshleifer/distilbart-cnn-12-6` (map-reduce) | Claude (Opus 5 default) with schema-validated structured outputs |
| Sector classification | `valhalla/distilbart-mnli-12-3` zero-shot | Per-sector direction + magnitude + rationale + opportunities/restrictions |
| Chronology / risks / FAQ | Regex act-mentions only | Policy background, timeline, risk flags, FAQ, persona impacts |
| Impact bullets | Heuristic — split summary into thirds | Real short/medium/long-term reasoning with risks per window |
| Cost | Free, runs offline | Pay-per-use API (Haiku 4.5 selectable for cheap runs) |
| Quality | Decent extractive feel | Genuinely 8th-grade friendly (measured: FK grade ≈ 7 on the demo) |

> "I built it dual-mode so the project works for **anyone** without an API key, but evaluators can plug in their own key and see professional-grade output. This is a real engineering choice, not laziness — it shows I can build production-ready pipelines that degrade gracefully."

### 1.9 Free-mode summarization — map-reduce
- **Map step**: split into ~900-word chunks, summarise each one with `distilbart-cnn` (max_length=220, min_length=80)
- **Reduce step**: concatenate the partial summaries and summarise them again
- This is the standard pattern for summarising documents longer than the model's context window

### 1.10 Free-mode classification — zero-shot
- `distilbart-mnli` is trained on Natural Language Inference, which lets it score arbitrary candidate labels against any input text — no fine-tuning needed
- I pass it 15 candidate sectors (Agriculture, Finance, Education, Healthcare, Technology, Environment, Defence, Labour, Energy, Infrastructure, Justice and Law, Social Welfare, Taxation, Trade and Commerce, Transport)
- `multi_label=True` because a Bill can affect several sectors at once
- Threshold = 0.2, top 5 returned

### 1.11 Claude-mode analysis (`ppiia_ai.py`)
- **Four structured stages**, each one `client.messages.parse()` call validated against a pydantic schema: (1) classify + metadata, (2) chronology, (3) impact + risks, (4) citizen summary
- Structured outputs constrain generation to the JSON schema — invalid JSON is impossible, so there's no regex JSON repair anywhere
- The bill text is sent as an identical `cache_control` block in all four calls, so stages 2–4 read it from Anthropic's **prompt cache** instead of re-billing the full document
- Closed vocabularies for category and sectors; every related act carries a `confidence` field (high = cited in the text, medium/low = model general knowledge) as the hallucination guardrail

### 1.12 Output schema (BillReport dataclass)
```python
{
  "title", "ministry", "objective", "summary",
  "sectors", "key_entities",
  "chronology",
  "impacts": {"short_term", "medium_term", "long_term"},
  "positives", "negatives", "risks"
}
```
A `dataclass` gives me type-checked attribute access in Python AND `to_dict()` for JSON serialisation in one place.

The Claude engine uses a richer **pydantic** schema on top (`schemas.py: BillAnalysis`) — pydantic validates and coerces LLM output at runtime, which a dataclass cannot, and it doubles as the structured-output contract sent to the API. `BillAnalysis.from_local()` adapts the local `BillReport`, so one dashboard renders both engines.

### 1.13 Deployment
- **Streamlit** (`app.py`) for local demos — tabs for Summary / Impacts / Chronology / Sectors / Risks / Raw JSON, with a download button
- **Gradio** (in the Colab notebook) for public sharing — `demo.launch(share=True)` gives me a public URL I can drop into a LinkedIn demo video
- **CLI** (`python ppiia.py path/to/bill.pdf`) for batch processing
- All three reuse the same `analyze()` function from `ppiia.py` — single source of truth

### 1.14 Conclusion
> "PPIIA is a complete, end-to-end NLP/LLM application: ingestion, cleaning, entity extraction, classification, summarization, structured output, dashboard, demo. It runs free on a laptop today and upgrades to a frontier LLM — Claude — the moment the user provides an API key. The code is modular, reproducible, tested, and has zero secrets in the repo."

### 1.15 Business use
- **Citizens** — instant briefing on new laws
- **Journalists & NGOs** — first-pass research tool
- **Startups / corporates** — quickly tell which Bills affect their sector
- **Educators** — teach civics with real, digestible content

---

## 2. Mock interview Q&A (the 10-mark section)

Read every answer aloud once.

### Why both Free mode and LLM mode? Why not just one?
> "Real engineering trade-off. Free mode means the project works on a laptop with zero setup — important for grading and reproducibility. LLM mode shows what's possible with a real instruction-following model. I detect the API key at runtime and switch transparently. It's the same pattern many open-source tools use, like LangChain or LlamaIndex."

### Why Claude for the LLM mode?
> "Three technical reasons. **Structured outputs**: `client.messages.parse()` takes my pydantic schema directly and the API constrains generation to it — the four stages literally cannot return malformed JSON. **Context**: Claude's 1M-token window fits any Indian bill whole, so there's no truncation or map-reduce in LLM mode. **Prompt caching**: I send the bill as an identical cached block in all four stage calls, so the document is billed roughly once, not four times. The UI also lets you drop to Haiku 4.5 when you want cheap fast runs."

### How do you guarantee valid JSON from the LLM?
> "Structured outputs. Each stage calls `client.messages.parse(..., output_format=MyPydanticModel)` — the API constrains generation to the model's JSON schema, and the SDK validates and returns a typed object via `response.parsed_output`. That's stronger than prompting for JSON and calling `json.loads()`, because required fields and closed enums (like severity = low/medium/high) are enforced at generation time, not hoped for."

### Why `distilbart-cnn-12-6` for summarization?
> "It's a **distilled** version of facebook/bart-large-cnn — about 3× smaller and 2× faster while keeping ~95% of the original ROUGE score. Small enough to run on CPU in a reasonable time. The full BART would be too slow for a free-tier demo."

### What is map-reduce summarization?
> "BART has a 1024-token context window, but Bills can be 50,000+ words. **Map step**: split into chunks that fit, summarise each chunk independently. **Reduce step**: concatenate the chunk summaries and summarise THAT into a final summary. It's the same pattern as MapReduce in big-data frameworks, applied to document summarization."

### Why not just feed the whole document to the LLM?
> "In **local mode** I can't — BART's context window is 1024 tokens, so map-reduce is mandatory. In **Claude mode** I do feed the whole document: the 1M-token context fits any real Indian bill, with a 400k-character cap purely as a cost guard. Instead of one giant do-everything prompt, I ask four focused questions over the same text — that keeps each answer sharp and lets each stage have its own strict output schema."

### What is zero-shot classification?
> "A classifier that can label text against arbitrary categories it has never been trained on, by formulating each label as a hypothesis and using a **Natural Language Inference** model to score whether the input *entails* the hypothesis. So I just hand it 'Agriculture, Finance, Healthcare, ...' as candidate labels at inference time — no fine-tuning, no labelled data."

### What is `multi_label=True` doing?
> "By default the classifier treats labels as mutually exclusive (softmax). With `multi_label=True`, each label is scored independently with a sigmoid — a Bill can be 'Healthcare AND Finance AND Technology' all at once. This is essential here because most Bills affect several sectors."

### Why regex for entity extraction instead of spaCy NER?
> "spaCy NER would give better person and location names, but it adds 50+ MB to the install and doesn't recognise Indian Bill conventions like 'The Aadhaar Act, 2016' very well. My three regexes (years, acts, ministries) are cheap, dependency-free, and tuned for the exact format Indian Bills use. If accuracy were critical I'd add a spaCy fallback."

### How does PDF extraction work?
> "`pdfplumber.open(path)` returns a list of `page` objects. Each page has `.extract_text()` which respects the PDF's text layer (NOT OCR — pdfplumber doesn't do OCR). I join the pages with newlines and feed the result into the cleaner. If `pdfplumber` isn't installed I fall back to `pypdf.PdfReader`. For scanned PDFs I'd need an OCR step like Tesseract — that's a known limitation."

### What's the difference between extractive and abstractive summarization?
> "**Extractive** picks sentences from the source verbatim (TextRank, BERTSum). **Abstractive** *rewrites* the content in new words (BART, T5, GPT). Abstractive is much more readable but can hallucinate. I use abstractive (BART or LLM) here because the brief explicitly says 'simple, 8th-grade-friendly summary' which extractive cannot produce."

### How do you handle hallucinations?
> "The system prompt explicitly says 'never invent facts not in the document'. I also constrain the output to a JSON schema, which makes it harder for the model to drift. For sectors I use a closed-vocabulary list. For longer documents I keep the LLM grounded by feeding it the cleaned text directly. There's no perfect fix — for production I'd add a retrieval-augmented step where each claim is checked against the source."

### What is `dataclass`? Why use it?
> "`@dataclass` is a Python decorator that auto-generates `__init__`, `__repr__`, `__eq__` etc. from typed class attributes. I use it for `BillReport` because I get type-checked attribute access in code (`report.sectors`) AND `asdict(report)` for clean JSON serialisation. Cleaner than plain dicts and lighter than Pydantic."

### Why store secrets in `.env`?
> "API keys must NEVER be committed to git. `.env` is read by `python-dotenv` at startup and added to `os.environ`. The repo includes `.env.example` as a template and `.gitignore` excludes the real `.env`. Standard 12-factor app practice."

### How would you deploy this to Hugging Face Spaces?
> "Create a Space with `streamlit` as the SDK, push `app.py` + `ppiia.py` + `requirements.txt`. Set `GROQ_API_KEY` as a Space secret in the settings. The Space will auto-build and serve the Streamlit app at a public URL. The brief lists this as a deliverable."

### What are the limitations?
> "1) No OCR — scanned PDFs return empty text. 2) Claude mode caps input at 400k characters (bigger than any real bill, but a cap nonetheless). 3) Free-mode impact analysis is heuristic, not real reasoning. 4) Sector list is hard-coded to 15 categories. 5) Chronology beyond the bill text is model knowledge, not a verified legislative database — which is why every related act carries a confidence label."

### How would you improve it?
> "1) **Add OCR** with Tesseract or paddleocr for scanned PDFs. 2) **RAG** — index a database of past Bills and retrieve related ones for the chronology section instead of relying on regex. 3) **Knowledge graph** of acts → amendments → committees, queried with NetworkX or Neo4j. 4) **Streaming output** in the Streamlit UI so the user sees the summary appearing token-by-token. 5) Add **human evaluation** of summaries with a thumbs-up/down feedback loop."

### Why JSON output instead of free-form text?
> "Structured output is essential for any app that displays the result in a UI. With JSON I can render each field in its own Streamlit tab. Free-form text would force me to parse the LLM's prose with regex, which is fragile."

### What are transformers in NLP?
> "Models built on the **self-attention** mechanism from the 2017 'Attention Is All You Need' paper. Each token can attend directly to every other token in the input, so the model captures long-range dependencies that RNNs/LSTMs struggle with. BART, BERT, GPT, T5, Llama — all transformer-based."

### What is the difference between BART and BERT?
> "**BERT** is encoder-only — designed for classification, NER, masked-language tasks. **BART** is encoder-decoder — designed for sequence-to-sequence tasks like summarization and translation. That's why I use BART (encoder-decoder) for summarization and BART-MNLI (which uses BART's encoder for classification) for zero-shot."

### What is a context window?
> "The maximum number of tokens a model can read in one forward pass. BART = 1024; Claude Opus 5 and Sonnet 5 = 1,000,000; Claude Haiku 4.5 = 200,000. If your input is longer, you must chunk or truncate — that's what map-reduce summarization is for in local mode. In Claude mode the window is big enough that whole bills fit."

### What is Langflow (mentioned in the brief)?
> "Langflow is a visual drag-and-drop UI for building LangChain pipelines. It generates the same kind of LLM chains I'm building in code, just visually. I chose to write Python directly because it's more reproducible, debuggable, and version-controllable than a JSON Langflow graph."

### How is this project different from projects 1 and 2?
> "**Project 1 (Music Clustering)** is **unsupervised classical ML** — KMeans on numeric audio features. **Project 2 (Toxicity)** is **supervised deep learning** — a BiLSTM classifying short text. **Project 3 (PPIIA)** is **NLP + LLM applications** — pipelining transformers and a frontier LLM (Claude) into a structured analyzer. Together they cover ML, DL, and LLMs."

### Walk me through what happens when I upload a PDF and click Analyze.
> "1) Streamlit writes the upload to a temp file and `extract_text_from_pdf()` runs pdfplumber. 2) The app builds a `SourceInfo` (name, characters, words). 3) The engine picker decides the path. 4) **Claude mode** → `analyze_with_claude()` runs `prepare_text()` (keeps clause structure, strips page furniture) then the four structured stages, and assembles a `BillAnalysis` with readability and timing metrics. 5) **Local mode** → `ppiia.analyze()` runs cleaning, regex entities, map-reduce summarization and zero-shot classification, and the flat report is adapted via `BillAnalysis.from_local()`. 6) One renderer draws the tabs — summary, sector chart, timeline, forecast, risks — plus the JSON and PDF download buttons."

### Why four LLM calls instead of one big prompt?
> "Focused prompts beat kitchen-sink prompts. Each stage asks one question and gets one strict schema back — classification can't bleed into the summary, and a failure in one stage doesn't corrupt the others. The bill text is identical across calls, so prompt caching means I'm not paying four times for the document. It also mirrors how you'd draw it in Langflow: four prompt→model→parser branches merging into one record — that mapping is written up in `docs/langflow_design.md`."

### How do you test an LLM app without spending money on every test run?
> "Dependency injection. `analyze_with_claude()` accepts an `engine` object; tests pass a `FakeEngine` whose `run_stage()` returns canned pydantic objects, so the whole assembly path — schema validation, metrics, coercion — is tested with zero network calls. 21 tests run in under 5 seconds. The real `ClaudeEngine` is a thin wrapper, so the untested surface is minimal."

### How do you measure summary quality?
> "Readability is the project's own success criterion — '8th-grade friendly'. Every analysis computes the Flesch-Kincaid grade and Flesch reading ease of the citizen summary with `textstat` and shows them in the metrics row; the bundled demo scores grade 7.0, under the 8th-grade target. Alongside that: processing time and a compression ratio (input words ÷ summary words)."

### What stops the model from inventing legislative history?
> "Three guardrails. The system prompt forbids inventing act names. The chronology schema forces a `confidence` field per related act — `high` only when the act is cited in the bill text, `medium/low` when it comes from general knowledge — and the UI explains that split to the user. And the dashboard carries a standing disclaimer to verify against sansad.in / PRS India. You can't make an LLM incapable of error, so the honest design is to label uncertainty instead of hiding it."

### What happens if the API call fails mid-analysis?
> "Every failure mode maps to a clear user-facing message: bad key → 'API key rejected', 429 → 'rate-limited, wait a minute', network down → 'check your connection', model refusal or truncated output → actionable text. The Anthropic SDK retries transient errors up to 3 times with backoff before I ever surface anything."

---

## 3. Last-minute checklist

- [ ] Run the Colab notebook end-to-end on a real Bill PDF (try one from sansad.in)
- [ ] Put your **ANTHROPIC_API_KEY** in `.env` so you can show Claude mode live — and remember the **⚡ Load sample demo** button works with no key at all
- [ ] Run `pytest` once so you can say "21 tests, all passing" with a straight face
- [ ] Record a 30-second demo video (Streamlit dashboard or the Gradio public URL) → upload to LinkedIn
- [ ] Push the project to GitHub (without `.env`!) — the brief makes this **mandatory**
- [ ] Have `ppiia.py`, `ppiia_ai.py`, `app.py` and this guide open in tabs during the live eval
- [ ] If asked "what would you do differently?" → OCR for scanned PDFs, RAG for chronology, streaming output
- [ ] If asked "how is this different from your other projects?" → ML vs DL vs LLM (see Q above)
