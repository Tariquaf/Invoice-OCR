import cv2
import pytesseract
import numpy as np
import frappe
import json
import re
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from frappe.utils.file_manager import get_file_path
from frappe.model.document import Document
from PIL import Image
from frappe.utils import add_days, get_url_to_form, nowdate


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
        text = ""

        def preprocess_image(pil_img):
            img = np.array(pil_img.convert("RGB"))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            scaled = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(scaled)
            thresh = cv2.adaptiveThreshold(
                enhanced, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 10
            )
            return thresh

        if file_path.endswith(".pdf"):
            images = convert_from_path(file_path, dpi=300)
            for img in images:
                processed = preprocess_image(img)
                text += pytesseract.image_to_string(processed, config="--psm 4 --oem 3 -l eng+urd")
        else:
            img = Image.open(file_path)
            processed = preprocess_image(img)
            text = pytesseract.image_to_string(processed, config="--psm 4 --oem 3 -l eng+urd")

        items = self.extract_items(text)
        extracted_data = {
            "items": items,
            "party": None
        }

        self.set("invoice_upload_item", [])
        for row in items:
            matched_item = frappe.db.get_value("Item", {"item_name": row["description"]})
            self.append("invoice_upload_item", {
                "ocr_description": row["description"],
                "qty": row["qty"],
                "rate": row["rate"],
                "item": matched_item
            })

        party_code = self.extract_party(text)
        if party_code:
            extracted_data["party"] = party_code.strip()

        self.extracted_data = json.dumps(extracted_data, indent=2)
        self.ocr_status = "Extracted"
        self.save()
        frappe.msgprint("OCR Extraction completed. Please review data before submitting.")

    def ensure_party_exists(self):
        extracted = json.loads(self.extracted_data or '{}')
        party = extracted.get("party")

        if not party or not party.strip():
            frappe.throw("Party is missing. Cannot create invoice.")

        if self.party_type == "Customer" and not frappe.db.exists("Customer", party):
            doc = frappe.get_doc({
                "doctype": "Customer",
                "customer_name": party.strip(),
                "customer_group": "All Customer Groups",
                "territory": "All Territories"
            })
            doc.insert(ignore_permissions=True)
            frappe.db.commit()
            extracted["party"] = doc.name

        elif self.party_type == "Supplier" and not frappe.db.exists("Supplier", party):
            doc = frappe.get_doc({
                "doctype": "Supplier",
                "supplier_name": party.strip(),
                "supplier_group": "All Supplier Groups",
                "country": "Pakistan"
            })
            doc.insert(ignore_permissions=True)
            frappe.db.commit()
            extracted["party"] = doc.name

        self.db_set("party", extracted["party"])
        self.extracted_data = json.dumps(extracted, indent=2)
        self.save()

    def create_invoice_from_child(self):
        extracted = json.loads(self.extracted_data or '{}')
        items = extracted.get("items", [])
        party = extracted.get("party")

        if not items:
            frappe.throw("No items found. Please extract first.")

        self.ensure_party_exists()

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
            item_code = frappe.db.get_value("Item", {"item_name": row["description"]})
            if not item_code:
                item_code = self.ensure_item_exists(row["description"])

            inv.append("items", {
                "item_code": item_code,
                "qty": row["qty"],
                "rate": row["rate"],
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

    def extract_items(self, text):
        items = []
        # Simplified and robust item extraction
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            # Look for quantity pattern
            qty_match = re.search(r'([\d,]+\.\d{3})\s*(kg|Units)', line)
            if qty_match:
                # Find description in previous lines
                description = ""
                j = i - 1
                while j >= 0 and j >= i - 3:
                    if lines[j].strip() and not re.search(r'QUANTITY|UNIT PRICE|AMOUNT|DESCRIPTION', lines[j], re.IGNORECASE):
                        description = lines[j].strip()
                        break
                    j -= 1
                
                # Find rate in current or next lines
                rate = None
                rate_match = re.search(r'([\d,]+\.\d{3})(?!.*(?:kg|Units))', line)
                if not rate_match:
                    # Check next 2 lines for rate
                    for k in range(i+1, min(i+3, len(lines))):
                        rate_match = re.search(r'([\d,]+\.\d{3})', lines[k])
                        if rate_match:
                            rate = rate_match.group(1)
                            break
                else:
                    rate = rate_match.group(1)
                
                if description and rate:
                    try:
                        items.append({
                            "description": description,
                            "qty": float(qty_match.group(1).replace(',', '')),
                            "rate": float(rate.replace(',', ''))
                        })
                    except:
                        pass
            i += 1
        return items

    def extract_party(self, text):
        # Simplified party extraction
        # 1. Look for Source field
        source_match = re.search(r'Source:\s*([^\n|]+)', text, re.IGNORECASE)
        if source_match:
            return source_match.group(1).strip()
        
        # 2. Look for Payment Communication
        payment_match = re.search(r'Payment\s+Communication:\s*([^\n]+)', text, re.IGNORECASE)
        if payment_match:
            return payment_match.group(1).strip()
        
        # 3. Look for Order number
        order_match = re.search(r'Order\s*[#:]\s*([^\s]+)', text, re.IGNORECASE)
        if order_match:
            return order_match.group(1).strip()
        
        # 4. Look for Invoice number pattern
        inv_match = re.search(r'Invoice\s+([A-Z]+/\d{4}/\d+)', text, re.IGNORECASE)
        if inv_match:
            return inv_match.group(1).strip()
        
        return None


@frappe.whitelist()
def extract_invoice(docname):
    doc = frappe.get_doc("Invoice Upload", docname)
    doc.extract_invoice()


@frappe.whitelist()
def debug_ocr_preview(docname):
    doc = frappe.get_doc("Invoice Upload", docname)
    file_path = get_file_path(doc.file)

    def preprocess_image(pil_img):
        img = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        scaled = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(scaled)
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 10
        )
        return thresh

    text = ""
    if file_path.endswith(".pdf"):
        images = convert_from_path(file_path, dpi=300)
        for img in images:
            processed = preprocess_image(img)
            text += pytesseract.image_to_string(processed, config="--psm 4 --oem 3 -l eng+urd")
    else:
        img = Image.open(file_path)
        processed = preprocess_image(img)
        text = pytesseract.image_to_string(processed, config="--psm 4 --oem 3 -l eng+urd")

    return text[:2000]  # Limit output to first 2000 characters
