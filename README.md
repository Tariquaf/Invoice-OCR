# 📄 Invoice OCR App for ERPNext v15 (Enhanced Fork)

This repository offers a significantly improved version of the Invoice OCR tool, originally created by [Mohtashim-1](https://github.com/Mohtashim-1/Invoice-OCR) and later enhanced by [Tariquaf](https://github.com/Tariquaf/Invoice-OCR). It automates data extraction from scanned PDF or image invoices and creates Sales or Purchase Invoices in ERPNext using OCR.

## 🚀 What's New in This Fork

- 🔁 Support for both Tesseract and optional PaddleOCR
- 🧾 Smarter parsing with improved line-item and metadata extraction
- 🌍 Multi-language OCR (including Urdu, Arabic, etc.)
- 🖼️ Advanced image preprocessing (deskewing, denoising, etc.)
- 📤 Flexible export options (JSON, CSV, or direct ERPNext integration)
- 🧱 Modular, extensible codebase with better logging and error handling

## 🧠 Core Logic Overview

The `invoice_upload-2.py` script defines a Frappe DocType `InvoiceUpload` that:

- Extracts text from uploaded PDFs or images using `pytesseract`
- Parses invoice data: number, date, totals, line-items
- Auto-creates draft Sales/Purchase Invoices in ERPNext
- Handles file processing errors with graceful logging

## 📂 ERPNext DocType: Invoice Upload

| Field            | Description                                |
|------------------|--------------------------------------------|
| Party Type       | Customer / Supplier                        |
| File             | Attach scanned invoice                     |
| OCR Status       | Pending / Processing / Extracted / Failed  |
| Extracted Data   | Raw JSON of OCR output                     |
| Create Invoice   | Triggers invoice generation in ERPNext     |

---

## ⚙️ Full Installation Guide

### ✅ 1. Prerequisites

Install required system packages:

```bash
sudo apt update
sudo apt install tesseract-ocr libtesseract-dev tesseract-ocr-eng tesseract-ocr-urd
sudo apt install poppler-utils  # For PDF processing
sudo apt install libgl1-mesa-glx  # For OpenCV

# Get the app from GitHub

[bench get-app https://github.com/Tariquaf/Invoice-OCR.git

# Activate your Frappe virtual environment

source ~/frappe-bench/env/bin/activate

# Install required Python libraries

pip install -r apps/invoice_ocr/requirements.txt

# Or manually install requirements

pip install opencv-python-headless pytesseract numpy PyPDF2 pdf2image Pillow requests

# Verify dependencies

python3 ~/frappe-bench/apps/invoice_ocr/verify_dep.py

# Deactivate virtual enviroment

deactivate

# 4. Install the app on your site
cd ~/frappe-bench
bench --site yoursite.com install-app invoice_ocr

#Apply necessary migrations
bench migrate

#Restart bench or supervisor
bench restart #for production
bench start #for development

#How to use
- From awsome bar, search for "New Invoice Upload"
- Select Customer or Supplier depending upon invoice type
- Click attach button and attach/select invoice
- A button "Extract from File" will appear on top
- Save and submit after verification. It will create a draft invoice and further amendments can be made in draft invoice.



