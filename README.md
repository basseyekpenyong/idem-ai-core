# idem-ai-core
Multi-language ASR data factory engine and dataset generation pipeline for Efik, Ibibio, and Yoruba, orchestrated automatically via voice or text prompts using the Aziz Agent.
========================================================================================
                               IDEMAI SYSTEM ARCHITECTURE
========================================================================================

    ┌────────────────────────────────────────────────────────────────────────┐
    │                        Dashboard UI & Voice Layer                      │
    │            (Streamlit / React Front-end + Web Audio Recorder)          │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │ Command / Voice Stream
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                               Aziz Agent                               │
    │     (Lightweight LLM / Intent Parser & Function Router Layer)          │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │ Triggers Workflow
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                      Universal ASR Pipeline Core                       │
    │    (FastAPI Backend / Handles Session Queues & Execution Paths)        │
    └───────┬───────────────────────────┬────────────────────────────┬───────┘
            │ Ingests                   │ Validates                  │ Merges
            ▼                           ▼                            ▼
    ┌───────────────┐           ┌───────────────┐            ┌───────────────┐
    │ Text Chunker  │           │ Audio Pipeline│            │ Data Factory  │
    │  & Scripting  │           │   (Librosa)   │            │   Manifest    │
    └───────┬───────┘           └───────┬───────┘            └───────┬───────┘
            │                           │                            │
            └───────────────────────────┼────────────────────────────┘
                                        ▼ 
                           ┌─────────────────────────┐
                           │ Language Profiles Rule  │
                           │   Cartridge Registry    │
                           │ (Efik, Ibibio, Yoruba)  │
                           └────────────┬────────────┘
                                        │ Loads Configuration
                                        ▼
                           ┌─────────────────────────┐
                           │   Target Storage Sync   │
                           │ (Google Drive / HF Sync)│
                           └─────────────────────────┘

========================================================================================
                      REPOSITORY & DIRECTORY STRUCTURE (idem-ai-core)
========================================================================================

idem-ai-core/
│
├── .github/workflows/         # CI/CD deployment automation (Linear integrations)
│
├── config/
│   ├── __init__.py
│   └── language_profiles.py   # Target language rules & alphabets (Efik, Ibibio, Yoruba)
│
├── engine/
│   ├── __init__.py
│   ├── validator.py          # Character whitelisting & strict text normalizers
│   ├── audio_pipeline.py     # Resampling (16kHz Mono WAV) & unique hash file mapping
│   └── script_generator.py   # Large document text chunker & mock translation database
│
├── agent/
│   ├── __init__.py
│   ├── aziz_orchestrator.py  # Aziz Agent: LLM tool caller & function router
│   └── voice_processor.py    # Mic audio stream capturing (Speech-to-Text orchestrator)
│
├── app/
│   └── dashboard.py          # Interactive UI client workspace (Streamlit front-end)
│
├── master_manifest.jsonl      # Aggregated production training dataset manifest 
├── requirements.txt           # Standard Python dependencies matrix
├── CONTRIBUTING.md            # LLM engineering constraints guide file
└── README.md                  # Project manual & runtime documentation
