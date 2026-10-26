# ATLAS — Local AI Document Parser

ATLAS is a Python-based document parsing and analysis application that uses a locally running AI model to extract information from documents and generate readable summaries.

The project combines document-processing libraries with Ollama and Gemma, with a Streamlit interface for interacting with the application.

## Features

- **Multi-format document support:** PDF, images, DOCX, TXT, and XLSX.
- **AI-powered extraction:** Extracts document text and identifies relevant fields and information.
- **Document summarization:** Generates summaries, important points, key details, dates, action items, and uncertainties.
- **Structured output:** Saves processing results in JSON format.
- **Interactive interface:** Upload and process documents through a Streamlit web application.
- **Local AI inference:** Uses Ollama to run the configured Gemma model locally.

## Tech Stack

- Python
- Ollama
- Gemma (`gemma4:e2b`)
- Streamlit
- PyMuPDF
- Pillow
- python-docx
- OpenPyXL
- JSON

## How It Works

1. The application loads a supported document.
2. Document-processing components extract text or prepare document pages for AI processing.
3. The local AI model helps structure the extracted information.
4. A summarization component generates a readable analysis.
5. The results can be viewed through the interface and saved as structured output.

## Running Locally

### Prerequisites

- Python installed
- Ollama installed and running
- The model configured by the application available in Ollama

### Setup

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Ensure the required Ollama model is available, then launch the interface:

```bash
python -m streamlit run app.py
```

## Project Status

This project is under development. Document extraction and local AI analysis have been tested, with results varying by document format and content. Spreadsheet analysis may require further refinement.

## Learning Goals

This project explores document-processing pipelines, local language models, structured AI outputs, and the integration of Python-based AI components into a user-facing application.
