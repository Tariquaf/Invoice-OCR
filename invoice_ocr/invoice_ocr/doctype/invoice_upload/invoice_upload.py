import cv2
import pytesseract
import numpy as np
import frappe
import json
import re
import traceback
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
        except Exception as e:
            frappe.db.set_value("Invoice Upload", self.name, "ocr_status", "Failed")
            frappe.db.commit()
            error_message = f"Invoice Creation Failed: {str(e)}\n{traceback.format_exc()}"
            frappe.log_error(error_message, "Invoice Creation Failed")
            frappe.throw(f"Invoice creation failed: {str(e)}")

    def extract_invoice(self):
        try:
            if not self.file:
                frappe.throw("No file attached.")

            file_path = get_file_path(self.file)
            text = ""

            # Enhanced Odoo-style preprocessing
            def preprocess_image(pil_img):
                try:
                    img = np.array(pil_img.convert("RGB"))
                    channels = img.shape[-1] if img.ndim == 3 else 1
                    
                    if channels == 3:
                        # Convert to grayscale
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = img
                        
                    # Enhance resolution (Odoo style)
                    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    
                    # Apply CLAHE for contrast enhancement (Odoo style)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(scaled)
                    
                    # Apply adaptive thresholding
                    thresh = cv2.adaptiveThreshold(
                        enhanced, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 15, 10
                    )
                    
                    # Apply erosion to reduce noise (Odoo style)
                    kernel = np.ones((3, 3), np.uint8)
                    processed = cv2.erode(thresh, kernel, iterations=1)
                    
                    return processed
                except Exception as e:
                    frappe.log_error(f"Image processing failed: {str(e)}", "OCR Error")
                    return pil_img  # Return original if processing fails

            if file_path.endswith(".pdf"):
                images = convert_from_path(file_path, dpi=300)
                for img in images:
                    processed = preprocess_image(img)
                    text += pytesseract.image_to_string(processed, config="--psm 4 --oem 3 -l eng+urd")
            else:
                img = Image.open(file_path)
                processed = preprocess_image(img)
                text = pytesseract.image_to_string(processed, config="--psm 4 --oem 3 -l eng+urd")

            # Save extracted text for debugging
            self.raw_ocr_text = text[:10000]  # Save first 10k characters
            self.save()
            
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
            
            return {
                "status": "success",
                "items": items,
                "party": party_code
            }
        except Exception as e:
            error_message = f"Extraction failed: {str(e)}\n{traceback.format_exc()}"
            frappe.log_error(error_message, "OCR Extraction Failed")
            frappe.throw(f"Extraction failed: {str(e)}")

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
        """Create item if it doesn't exist, handling special characters"""
        # Clean description to remove special characters
        cleaned_description = re.sub(r'[^\w\s\.\-\(\)\/]', '', description)[:140]
        
        # Try to find existing item by cleaned name
        item_code = frappe.db.get_value("Item", {"item_name": cleaned_description})
        
        if not item_code:
            # Create a safe item code
            safe_item_code = re.sub(r'[^\w\-]', '_', cleaned_description)[:130]
            
            # Ensure unique item code
            base_code = safe_item_code
            counter = 1
            while frappe.db.exists("Item", safe_item_code):
                safe_item_code = f"{base_code}_{counter}"
                counter += 1
            
            item = frappe.get_doc({
                "doctype": "Item",
                "item_name": cleaned_description,
                "item_code": safe_item_code,
                "item_group": "All Item Groups",
                "stock_uom": "Nos",
                "is_stock_item": 0
            })
            item.insert(ignore_permissions=True)
            item_code = item.name
        
        return item_code

    def extract_items(self, text):
        items = []
        # Look for quantity patterns in the text
        qty_matches = re.finditer(r'(\d+,\d+\.\d{3}|\d+\.\d{3})\s*(kg|Units)', text, re.IGNORECASE)
        
        for match in qty_matches:
            try:
                qty_str = match.group(1).replace(',', '')
                unit = match.group(2)
                qty = float(qty_str)
                
                # Find description in previous lines
                desc_start = text.rfind('\n', 0, match.start()) + 1
                desc_end = match.start()
                description = text[desc_start:desc_end].strip()
                
                # Clean up description
                description = re.sub(r'^\W+|\W+$', '', description)  # Remove surrounding symbols
                description = re.sub(r'\s+', ' ', description)  # Collapse multiple spaces
                
                # Find rate in the same line or next
                rate_match = re.search(r'(\d+,\d+\.\d{3}|\d+\.\d{3})(?!.*(?:kg|Units))', 
                                      text[match.start():match.start()+100])
                rate = float(rate_match.group(1).replace(',', '')) if rate_match else 0.0
                
                if description:
                    items.append({
                        "description": description,
                        "qty": qty,
                        "rate": rate
                    })
            except Exception as e:
                frappe.log_error(f"Item extraction failed: {str(e)}", "Item Extraction Error")
                continue
        
        # If no items found, fallback to original method
        if not items:
            lines = text.splitlines()
            pattern = re.compile(r"(.+?)\s+(\d+\.\d{1,2})\s+(\d+\.\d{1,2})\s+\$?(\d+\.\d{1,2})")
            for line in lines:
                match = pattern.search(line)
                if match:
                    try:
                        description = match.group(1).strip()
                        qty = float(match.group(2))
                        rate = float(match.group(3))
                        items.append({
                            "description": description,
                            "qty": qty,
                            "rate": rate
                        })
                    except:
                        continue
        
        return items

    def extract_party(self, text):
        # Try to extract from Source field
        source_match = re.search(r'Source:\s*([^\n|]+)', text)
        if source_match:
            return source_match.group(1).strip()
        
        # Try to extract from Payment Communication
        payment_match = re.search(r'Payment\s+Communication:\s*([^\n]+)', text)
        if payment_match:
            return payment_match.group(1).strip()
            
        # Try to extract from Order number
        order_match = re.search(r'Order\s*[#:]\s*([^\s]+)', text, re.IGNORECASE)
        if order_match:
            return order_match.group(1).strip()
            
        # Fallback to original method
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if any(key.lower() in line.lower() for key in ["Customer Code", "Supplier Code", "Customer:", "Supplier:"]):
                parts = line.split(":")
                if len(parts) > 1 and parts[1].strip():
                    return parts[1].strip()
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line:
                        return next_line
        return None


@frappe.whitelist()
def extract_invoice(docname):
    try:
        doc = frappe.get_doc("Invoice Upload", docname)
        result = doc.extract_invoice()
        return result
    except Exception as e:
        frappe.log_error(f"Extract invoice failed: {str(e)}", "Extract Invoice Error")
        return {"status": "error", "message": str(e)}


# Debug method to test OCR safely
@frappe.whitelist()
def debug_ocr_preview(docname):
    try:
        doc = frappe.get_doc("Invoice Upload", docname)
        file_path = get_file_path(doc.file)

        def preprocess_image(pil_img):
            try:
                img = np.array(pil_img.convert("RGB"))
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(scaled)
                thresh = cv2.adaptiveThreshold(
                    enhanced, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 15, 10
                )
                kernel = np.ones((3, 3), np.uint8)
                processed = cv2.erode(thresh, kernel, iterations=1)
                return processed
            except Exception as e:
                frappe.log_error(f"Debug image processing failed: {str(e)}", "OCR Debug Error")
                return pil_img

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

        # Save to document for debugging
        doc.raw_ocr_text = text[:10000]
        doc.save()
        
        return text[:5000]  # Limit output to first 5000 characters
    except Exception as e:
        frappe.log_error(f"OCR debug failed: {str(e)}", "OCR Debug Error")
        return f"Error: {str(e)}"
