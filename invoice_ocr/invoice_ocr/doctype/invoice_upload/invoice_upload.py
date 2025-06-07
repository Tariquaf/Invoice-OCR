import cv2
import pytesseract
import numpy as np
import frappe
import json
import re
import base64
import io
import traceback
import requests
import uuid
from PyPDF2 import PdfReader
from pdf2image import convert_from_path, convert_from_bytes
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

        file_path = get_file_path(self.file)
        file_content = frappe.get_doc("File", {"file_url": self.file}).get_content()

        # Extract text with OCR
        text = self._parse_attachment(file_content)
        if not text:
            frappe.throw("Failed to extract text from the document.")

        # Extract structured data with AI
        extracted_data = self.extract_values_with_ai(text)
        if not extracted_data:
            frappe.throw("Failed to extract structured data from document.")

        # Process extracted data
        self.process_extracted_data(extracted_data)

    def process_extracted_data(self, extracted_data):
        # Set party
        party_code = extracted_data.get("party", {}).get("name")
        if party_code:
            self.ensure_party_exists(party_code)
        
        # Set items
        self.set("invoice_upload_item", [])
        for row in extracted_data.get("invoice_lines", []):
            description = row.get("product", "")
            qty = row.get("quantity", 0)
            rate = row.get("price_unit", 0)
            
            matched_item = frappe.db.get_value("Item", {"item_name": description})
            
            self.append("invoice_upload_item", {
                "ocr_description": description,
                "qty": qty,
                "rate": rate,
                "item": matched_item
            })

        # Set dates
        if extracted_data.get("invoice_date"):
            try:
                self.posting_date = datetime.strptime(extracted_data["invoice_date"], "%Y-%m-%d").date()
            except:
                pass

        self.extracted_data = json.dumps(extracted_data, indent=2)
        self.ocr_status = "Extracted"
        self.save()
        frappe.msgprint("OCR Extraction completed. Please review data before submitting.")

    def ensure_party_exists(self, party_name):
        if not party_name or not party_name.strip():
            frappe.throw("Party name is missing. Cannot create invoice.")

        if self.party_type == "Customer" and not frappe.db.exists("Customer", party_name):
            doc = frappe.get_doc({
                "doctype": "Customer",
                "customer_name": party_name.strip(),
                "customer_group": "All Customer Groups",
                "territory": "All Territories"
            })
            doc.insert(ignore_permissions=True)
            frappe.db.commit()
            self.party = doc.name

        elif self.party_type == "Supplier" and not frappe.db.exists("Supplier", party_name):
            doc = frappe.get_doc({
                "doctype": "Supplier",
                "supplier_name": party_name.strip(),
                "supplier_group": "All Supplier Groups",
                "country": "Pakistan"
            })
            doc.insert(ignore_permissions=True)
            frappe.db.commit()
            self.party = doc.name
        else:
            self.party = party_name

        self.save()

    def create_invoice_from_child(self):
        extracted = json.loads(self.extracted_data or '{}')
        items = extracted.get("invoice_lines", [])
        party = self.party

        if not items:
            frappe.throw("No items found. Please extract first.")

        if not self.party:
            frappe.throw("Unable to determine or create party.")

        if self.party_type == "Supplier":
            inv = frappe.new_doc("Purchase Invoice")
            inv.supplier = self.party
        else:
            inv = frappe.new_doc("Sales Invoice")
            inv.customer = self.party

        expense_account = self.get_expense_account()

        for row in items:
            description = row.get("product", "")
            qty = row.get("quantity", 0)
            rate = row.get("price_unit", 0)
            
            item_code = frappe.db.get_value("Item", {"item_name": description})
            if not item_code:
                item_code = self.ensure_item_exists(description)

            inv.append("items", {
                "item_code": item_code,
                "qty": qty,
                "rate": rate,
                "uom": "Nos",
                "expense_account": expense_account
            })

        posting_date = getattr(self, "posting_date", None) or nowdate()
        inv.posting_date = posting_date
        inv.due_date = add_days(posting_date, 30)
        inv.insert(ignore_permissions=True)

        frappe.msgprint(f"<a href='{get_url_to_form(inv.doctype, inv.name)}'>{inv.name}</a> created")

    def get_expense_account(self):
        company = frappe.defaults.get_user_default("Company")
        account = frappe.db.get_value("Company", company, "default_expense_account")
        if not account:
            account = frappe.db.get_value("Account", {
                "account_type": "Expense",
                "company": company,
                "is_group": 0
            }, "name")
        if not account:
            frappe.throw("No default Expense Account found for the company.")
        return account

    def ensure_item_exists(self, description):
        item_code = frappe.db.get_value("Item", {"item_name": description})
        if not item_code:
            item = frappe.get_doc({
                "doctype": "Item",
                "item_name": description,
                "item_code": description,
                "item_group": "All Item Groups",
                "stock_uom": "Nos",
                "is_stock_item": 0
            })
            item.insert(ignore_permissions=True)
            item_code = item.name
        return item_code

    def _parse_attachment(self, file_content):
        """Parse file content and return extracted text"""
        try:
            # Try to read as PDF first
            try:
                reader = PdfReader(io.BytesIO(file_content))
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                if text.strip():
                    return text
            except:
                pass

            # If PDF extraction failed, try OCR
            try:
                # Convert PDF to images
                images = convert_from_bytes(file_content)
                text = ""
                for img in images:
                    processed = self._preprocess_image(img)
                    text += pytesseract.image_to_string(processed, config="--psm 4 --oem 3 -l eng+urd")
                return text
            except:
                # Handle image files directly
                try:
                    img = Image.open(io.BytesIO(file_content))
                    processed = self._preprocess_image(img)
                    return pytesseract.image_to_string(processed, config="--psm 4 --oem 3 -l eng+urd")
                except:
                    frappe.log_error("Failed to parse file content")
                    return ""
        except Exception as e:
            frappe.log_error(f"Error parsing attachment: {str(e)}")
            return ""

    def _preprocess_image(self, img):
        """Enhanced image preprocessing"""
        try:
            img_array = np.asarray(img)
            channels = img_array.shape[-1] if img_array.ndim == 3 else 1

            if channels == 3:
                # Convert to grayscale
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Enhance resolution
            scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(scaled)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                enhanced, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 10
            )
            
            # Apply erosion to reduce noise
            kernel = np.ones((3, 3), np.uint8)
            processed = cv2.erode(thresh, kernel, iterations=1)
            
            return processed
        except:
            return img

    def extract_values_with_ai(self, text):
        """Extract structured data using AI approach"""
        try:
            # Prepare system prompt
            system_prompt = self._ai_system_prompt()
            
            # Prepare user prompt
            user_prompt = self._ai_user_prompt(text)
            
            # Call AI service
            response = self._make_ai_request(system_prompt, user_prompt)
            
            # Parse and return response
            return json.loads(response)
        except Exception as e:
            frappe.log_error(f"AI extraction failed: {str(e)}\n{traceback.format_exc()}")
            return None

    def _ai_system_prompt(self):
        """Generate system prompt for AI"""
        party_type = "vendor" if self.party_type == "Supplier" else "customer"
        document_type = "invoice" if self.party_type == "Customer" else "bill"
        
        return f"""
        You are an invoice digitizer. Extract values from the provided document and return valid JSON.
        The document is a {party_type} {document_type} for {frappe.db.get_value("Company", frappe.defaults.get_user_default("Company"), "company_name")}.
        Extract the following information:
        - Partner details (name, vat_id, email)
        - Invoice number, date, due date
        - Line items (product, quantity, price_unit, discount, tax_rate)
        - Notes
        
        Return only a valid JSON object with this structure:
        {{
            "partner": {{
                "name": "String",
                "vat_id": "String",
                "email": "String"
            }},
            "invoice_number": "String",
            "invoice_date": "YYYY-MM-DD",
            "due_date": "YYYY-MM-DD",
            "invoice_lines": [
                {{
                    "product": "String",
                    "quantity": Float,
                    "price_unit": Float,
                    "discount": Float,
                    "tax_rate": Float
                }}
            ],
            "notes": "String"
        }}
        """

    def _ai_user_prompt(self, text):
        """Prepare user prompt for AI"""
        return f"""
        Extract information from this document:
        
        {text[:5000]}  # Truncate to avoid token limits
        """

    def _make_ai_request(self, system_prompt, user_prompt):
        """Make request to AI service (placeholder implementation)"""
        # In a real implementation, this would call an AI API
        # For now, we'll simulate with a regex-based fallback
        
        # Fallback to regex extraction if AI service is not available
        return json.dumps(self.fallback_extraction(user_prompt))

    def fallback_extraction(self, text):
        """Fallback extraction method when AI is not available"""
        # Extract party name
        party_name = None
        party_matches = re.findall(r'(?:Customer|Vendor|Supplier|Buyer|Client):?\s*([^\n]+)', text, re.IGNORECASE)
        if party_matches:
            party_name = party_matches[0].strip()
        
        # Extract items
        items = []
        # Look for quantity patterns
        qty_matches = re.finditer(r'(\d+\.\d+)\s*(kg|Units|PCS|pcs|kg|Kg)', text)
        for match in qty_matches:
            qty = float(match.group(1))
            unit = match.group(2)
            
            # Find description in previous lines
            lines = text.splitlines()
            line_index = text[:match.start()].count('\n')
            description = ""
            for i in range(max(0, line_index-3), line_index):
                if lines[i].strip() and not re.search(r'QUANTITY|PRICE|AMOUNT|TOTAL', lines[i], re.IGNORECASE):
                    description = lines[i].strip()
                    break
            
            # Find price in surrounding area
            price = None
            price_match = re.search(r'(\d+\.\d{2,3})', text[match.end():match.end()+100])
            if price_match:
                price = float(price_match.group(1))
            
            if description and price:
                items.append({
                    "product": description,
                    "quantity": qty,
                    "price_unit": price
                })
        
        # Extract dates
        dates = re.findall(r'\d{2,4}[-/]\d{1,2}[-/]\d{1,4}', text)
        invoice_date = dates[0] if dates else None
        due_date = dates[1] if len(dates) > 1 else None
        
        return {
            "partner": {
                "name": party_name or "Unknown"
            },
            "invoice_lines": items,
            "invoice_date": invoice_date,
            "due_date": due_date
        }


@frappe.whitelist()
def extract_invoice(docname):
    doc = frappe.get_doc("Invoice Upload", docname)
    doc.extract_invoice()


@frappe.whitelist()
def debug_ocr_preview(docname):
    doc = frappe.get_doc("Invoice Upload", docname)
    file_path = get_file_path(doc.file)
    file_content = frappe.get_doc("File", {"file_url": doc.file}).get_content()
    
    # Use the new parsing method
    text = doc._parse_attachment(file_content)
    return text[:2000]  # Limit output
