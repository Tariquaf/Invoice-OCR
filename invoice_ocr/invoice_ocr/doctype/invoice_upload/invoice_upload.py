import cv2
import pytesseract
import numpy as np
import frappe
import json
import re
import io
import difflib
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from frappe.utils.file_manager import get_file_path
from frappe.model.document import Document
from PIL import Image
from frappe.utils import add_days, get_url_to_form, nowdate
from datetime import datetime

class InvoiceUpload(Document):
    def on_submit(self):
        try:
            self.reload()
            self.create_invoice_from_child()
        except Exception:
            frappe.db.set_value("Invoice Upload", self.name, "ocr_status", "Failed")
            frappe.db.commit()
            frappe.log_error(frappe.get_traceback(), "Invoice Creation Failed")
            raise

    def extract_invoice(self):
        if not self.file:
            frappe.throw("No file attached.")

        file_content = frappe.get_doc("File", {"file_url": self.file}).get_content()
        text = self._extract_text_from_file(file_content)

        if not text:
            frappe.throw("Failed to extract text from the document.")

        extracted_data = self._extract_invoice_data(text)
        self._process_extracted_data(extracted_data)

        frappe.msgprint("Invoice data extracted successfully. Please review before submitting.")

    def _extract_text_from_file(self, file_content):
        try:
            try:
                reader = PdfReader(io.BytesIO(file_content))
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
                if text.strip():
                    return text
            except:
                pass

            images = []
            try:
                images = convert_from_bytes(file_content)
            except:
                try:
                    img = Image.open(io.BytesIO(file_content))
                    images = [img]
                except:
                    pass

            text = ""
            for img in images:
                processed_img = self._preprocess_image(img)
                text += pytesseract.image_to_string(processed_img, config="--psm 6 -l eng+urd")

            return text if text.strip() else None

        except Exception as e:
            frappe.log_error(f"Text extraction error: {str(e)}")
            return None

    def _preprocess_image(self, img):
        try:
            img_array = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            _, threshold = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return threshold
        except:
            return img

    def _extract_invoice_data(self, text):
        text = self._normalize_text(text)

        data = {
            "partner": {
                "name": self._extract_party_name(text)
            },
            "invoice_number": self._extract_invoice_number(text),
            "invoice_lines": self._extract_invoice_lines(text),
            "invoice_date": self._extract_date(text),
            "due_date": None
        }
        return data

    def _normalize_text(self, text):
        text = re.sub(r"[^\x00-\x7F؀-ۿ]+", " ", text)  # Retain ASCII + Urdu
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_party_name(self, text):
        own_names = [frappe.get_single("Global Defaults").default_company]

        party_patterns = [
            r"(?i)(Bill\s*To|Sold\s*To|Client|Customer|Buyer|Invoiced To)[:\-]?\s*(.*)",
            r"(?i)(Bill\s*From|Vendor|Supplier|Partner|From|Payable To)[:\-]?\s*(.*)",
            r"(?i)(Company|Co\.?|Ltd|Pvt\.?|Mill|Corporation)\s+(.*)"
        ]

        lines = text.splitlines()
        candidates = []

        for line in lines:
            for pattern in party_patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(2).strip()
                    if len(name) > 2 and not self._is_own_company(name, own_names):
                        candidates.append(name)

        if candidates:
            return candidates[0]

        # Check header/footer
        lines_to_check = lines[:10] + lines[-10:]
        for line in lines_to_check:
            name = line.strip()
            if not self._is_own_company(name, own_names) and re.search(r"[A-Z][a-z].*(Ltd|Pvt|Mill|Company|Corp)", name, re.IGNORECASE):
                return name

        return "Unknown"

    def _is_own_company(self, name, own_names):
        for own in own_names:
            ratio = difflib.SequenceMatcher(None, name.lower(), own.lower()).ratio()
            if ratio > 0.85:
                return True
        return False

    def _extract_invoice_number(self, text):
        patterns = [
            r"(?i)(Invoice No\.?|Invoice Number|INV\s*#?|Bill No\.?|Bill Number|Invoice)\s*[:\-]?\s*([\w\-\/]+)",
            r"(?i)\bINV[-_ ]?(\d+)\b"
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(2).strip() if len(match.groups()) > 1 else match.group(1).strip()
        return None

    def _extract_invoice_lines(self, text):
        lines = []
        quantity_matches = re.finditer(r"(\d+\.?\d*)\s*(kg|units|pcs|bags|Nos)?\b", text, re.IGNORECASE)

        for match in quantity_matches:
            qty = float(match.group(1))
            unit = match.group(2) or "Nos"
            desc = self._find_description(text, match.start())
            price = self._find_price(text, match.end())

            if desc and price:
                lines.append({
                    "product": desc,
                    "quantity": qty,
                    "price_unit": price
                })

        return lines

    def _find_description(self, text, pos):
        lines = text[:pos].splitlines()
        for line in reversed(lines[-3:]):
            clean_line = line.strip()
            if clean_line and not re.search(r"quantity|price|amount|total", clean_line, re.IGNORECASE):
                return clean_line
        return None

    def _find_price(self, text, pos):
        nearby_text = text[pos:pos+100]
        match = re.search(r"(\d+\.\d{2,3})", nearby_text)
        return float(match.group(1)) if match else 0.0

    def _extract_date(self, text):
        date_patterns = [
            r"(?i)(Date|Invoice Date|Dated)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(2) if len(match.groups()) > 1 else match.group(1)
                for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
                    try:
                        return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
                    except:
                        continue
        return None

    def _process_extracted_data(self, data):
        if data["partner"]["name"] and data["partner"]["name"] != "Unknown":
            self.ensure_party_exists(data["partner"]["name"])

        self.set("invoice_upload_item", [])

        for item in data["invoice_lines"]:
            self.append("invoice_upload_item", {
                "ocr_description": item["product"],
                "qty": item["quantity"],
                "rate": item["price_unit"],
                "item": self._get_matching_item(item["product"])
            })

        if data.get("invoice_date"):
            self.posting_date = data["invoice_date"]
        if data.get("invoice_number"):
            self.invoice_number = data["invoice_number"]

        self.extracted_data = json.dumps(data, indent=2)
        self.ocr_status = "Extracted"
        self.save()

    def _get_matching_item(self, description):
        item_code = frappe.db.get_value("Item", {"item_name": description})
        if not item_code:
            item_code = self.ensure_item_exists(description)
        return item_code

    # Note: ensure_party_exists, create_invoice_from_child, get_expense_account, ensure_item_exists
    # should be implemented separately as needed


@frappe.whitelist()
def extract_invoice(docname):
    doc = frappe.get_doc("Invoice Upload", docname)
    doc.extract_invoice()


@frappe.whitelist()
def debug_ocr_preview(docname):
    doc = frappe.get_doc("Invoice Upload", docname)
    file_content = frappe.get_doc("File", {"file_url": doc.file}).get_content()
    text = doc._extract_text_from_file(file_content)
    return text[:2000] if text else "No text extracted"
