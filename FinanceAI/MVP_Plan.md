# FinanceAI - Minimum Viable Product (MVP) Plan: AI for AP Automation

This plan outlines the development of the Minimum Viable Product (MVP) for FinanceAI, focusing on automating Accounts Payable (AP) workflows for non-profit organizations within a 3-month timeframe and a budget of $6,000.

## 1. Goals

*   Deliver a functional SaaS product automating the core AP workflow (invoice intake, data extraction, basic validation, approval routing, data export/integration) within 3 months.
*   Provide a clean, polished, and intuitive user experience from the start.
*   Validate the core value proposition with initial non-profit clients (handling ~100-200 invoices/month).
*   Establish a scalable architecture foundation for future expansion (HR, fundraising, advanced accounting, exception handling).

## 2. MVP Scope (Phase 1 Features - 3 Month Delivery)

*   **Invoice Intake:**
    *   Web-based interface for manual PDF invoice uploads.
    *   Mechanism to receive invoices via a dedicated email address.
*   **Data Extraction (OCR + LLM):**
    *   Integration with **Azure Form Recognizer** for OCR.
    *   Integration with **Azure OpenAI Service (ChatGPT models)** for key field extraction: Vendor Name, Invoice Date, Due Date, Invoice Amount, PO Number.
*   **Basic Validation:**
    *   Duplicate invoice detection.
    *   PO matching (Requires mechanism for PO data availability - e.g., simple upload/list).
    *   Flagging interface for identified issues.
*   **Approval Workflow:**
    *   Simple, configurable approval routing logic.
    *   Interface for approvers to view invoice data and approve/reject.
    *   Logging of actions for a basic audit trail.
*   **Data Export & Integration:**
    *   Functionality to export processed data to standard CSV.
    *   *Target:* API integration POC (Proof of Concept) or initial implementation for **one** key system (e.g., QuickBooks or Sage Intacct) based on API complexity assessment within the timeframe.
    *   Research and preparation for integrating with other target systems (Business Central, Abila MIP) post-MVP or via robust CSV export templates.
*   **User Interface (UI):**
    *   Secure user authentication & basic role management.
    *   **Polished, clean, and intuitive interface:**
        *   Dashboard for invoice status tracking.
        *   Clear invoice detail view.
        *   User-friendly document uploader.

## 3. Technology Stack

*   **Frontend:** React / Vue.js (To be finalized based on preference/efficiency).
*   **Backend:** Python (Django/Flask recommended) / Node.js.
*   **Database:** PostgreSQL.
*   **AI / OCR:**
    *   **Primary:** Microsoft Azure Cloud Platform
        *   OCR: **Azure Form Recognizer**
        *   LLM: **Azure OpenAI Service (ChatGPT)**
    *   *Alternatives:* AWS/Claude, Google Cloud/Gemini (if primary faces unexpected roadblocks).
*   **Deployment:** Docker, Azure App Service / Azure Functions / AKS (based on final architecture).
*   **Storage:** Azure Blob Storage.

## 4. High-Level Architecture

*   **Frontend:** SPA communicating via APIs.
*   **Backend:** RESTful API on Azure.
*   **Processing:** Asynchronous task queue (e.g., Azure Queue Storage + Azure Functions, or Celery with Redis on VM/AKS) for OCR and LLM processing to ensure scalability.
*   **Storage:** Secure Azure Blob Storage for invoice documents.
*   *Design Consideration:* Modular design to facilitate future agent additions (e.g., exception handling) and integrations (HR, fundraising).

## 5. Development Timeline (3 Months)

*   **Month 1: Foundation & Core Extraction**
    *   Finalize Frontend/Backend framework choice.
    *   Azure infrastructure setup (App Service/Functions, DB, Storage, OpenAI).
    *   CI/CD pipeline basics.
    *   Database schema & User authentication.
    *   Basic UI shell & PDF upload component.
    *   Integrate Azure Form Recognizer (OCR) & Azure OpenAI (LLM) for initial data extraction.
    *   Basic dashboard UI setup.
*   **Month 2: Workflow & Validation**
    *   Refine data extraction accuracy.
    *   Implement duplicate checking & PO matching logic.
    *   Build approval workflow engine & associated UI screens.
    *   Implement audit trail logging.
    *   Develop email intake mechanism.
    *   Focus on UI polish and usability testing.
*   **Month 3: Integration, Export & Polish**
    *   Implement robust CSV export.
    *   Develop API integration POC/implementation for **one** target accounting system.
    *   Intensive end-to-end testing & bug fixing.
    *   Security review and hardening.
    *   Final UI/UX refinements based on feedback.
    *   Deployment preparation & documentation.

## 6. Team (Based on Project Description)

*   **Technical Lead (Abdul Wahab Awan):** Design, development, potential subcontractor management.
*   *(Optional)* UI/UX Designer (Consultation or short contract for initial polish).

## 7. Budget

*   **MVP Development:** $6,000 (as agreed)
*   **Note:** This excludes ongoing cloud hosting and API usage costs (expected to be low initially based on volume).

## 8. Key Assumptions

*   Availability of clear API documentation and test environments for target accounting systems.
*   A mechanism for providing PO data (e.g., CSV upload) will be defined for MVP matching.
*   Approval logic requirements are relatively simple for the MVP. 