# FinanceAI - Minimum Viable Product (MVP) Plan: AI for AP Automation

This plan outlines the development of the Minimum Viable Product (MVP) for FinanceAI, focusing on automating Accounts Payable (AP) workflows for non-profit organizations within a 3-month timeframe and a budget of $6,000.

## 1. Goals

*   Deliver a functional SaaS product automating the core AP workflow (invoice intake, data extraction, basic validation, approval routing, data export/integration) within 3 months.
*   Provide a clean, polished, and intuitive user experience from the start.
*   Validate the core value proposition with initial non-profit clients (handling ~100-200 invoices/month).
*   Establish a scalable architecture foundation for future expansion (HR, fundraising, advanced accounting, exception handling).

## 2. MVP Scope

*   **Invoice Intake:**
    *   Web-based interface for manual PDF invoice uploads.
    *   Mechanism to receive invoices via a dedicated email address.
*   **Data Extraction (OCR + LLM):**
    *   Integration with **AWS Textract** for OCR.
    *   Integration with LLM (**AWS Bedrock/Claude** or **OpenAI API**) for key field extraction: Vendor Name, Invoice Date, Due Date, Invoice Amount, PO Number.
*   **Basic Validation:**
    *   Duplicate invoice detection.
    *   PO matching (Includes MVP implementation of a simple mechanism for PO data upload/management).
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
*   **Basic Error Handling:**
    *   Mechanism to flag processing errors (e.g., OCR/LLM failures, unparseable emails) for user review within the dashboard.

## 3. Technology Stack

*   **Frontend:** Next.js (React-based).
*   **Backend:** Python with Django (preferred for its robust ORM, admin interface, and security features).
*   **Database:** PostgreSQL on AWS RDS.
*   **AI / OCR:**
    *   **Primary:** Amazon Web Services (AWS)
        *   OCR: **AWS Textract**
        *   LLM: **AWS Bedrock with Claude** (primary) or **OpenAI API** (alternative)
    *   *Note:* Both LLM options will be evaluated during development to determine the best fit for accuracy and cost.
*   **Deployment:** 
    *   **Primary:** AWS App Runner (for simplified deployment and scaling)
    *   *Alternative:* AWS ECS with Fargate (if more control is needed)
*   **Processing:** AWS SQS + AWS Lambda for asynchronous task processing (OCR and LLM operations).
*   **Storage:** AWS S3 for secure document storage.

## 4. High-Level Architecture

*   **Frontend Layer:**
    *   Next.js SPA communicating with backend via RESTful APIs.
    *   Responsive design optimized for desktop and tablet use.
    *   Client-side state management with React Context or Redux.
    *   Secure authentication flow with JWT tokens.

*   **Backend Layer:**
    *   Django REST Framework API on AWS App Runner.
    *   Role-based access control (Admin, Uploader, Approver).
    *   API endpoints for invoice management, user management, and system configuration.
    *   Integration with external services (OCR, LLM, accounting systems).

*   **Processing Layer:**
    *   Asynchronous task queue using AWS SQS.
    *   AWS Lambda functions triggered by SQS messages for:
        *   OCR processing with AWS Textract.
        *   LLM processing with AWS Bedrock/Claude or OpenAI API.
        *   Email parsing and attachment extraction.
    *   Error handling and retry mechanisms.

*   **Data Layer:**
    *   PostgreSQL database on AWS RDS with the following core tables:
        *   Users (authentication, roles, preferences)
        *   Organizations (client information)
        *   Invoices (extracted data, status, metadata)
        *   PurchaseOrders (for matching)
        *   ApprovalWorkflows (configurations)
        *   ApprovalActions (audit trail)
        *   SystemLogs (error tracking, performance metrics)
    *   AWS S3 buckets for:
        *   Original invoice documents
        *   Processed data exports
        *   System backups

*   **Integration Layer:**
    *   API clients for accounting systems (initially QuickBooks or Sage Intacct).
    *   Email service integration for invoice intake.
    *   Export functionality for CSV and accounting system formats.

*   **Design Considerations:**
    *   Modular architecture to facilitate future agent additions (e.g., exception handling).
    *   Scalable design to accommodate growth beyond MVP volume.
    *   Security-first approach with encryption at rest and in transit.
    *   Comprehensive logging for troubleshooting and audit purposes.

## 5. Development Timeline (3 Months)

*   **Month 1: Foundation & Core Extraction**
    *   Finalize Backend framework choice (Django).
    *   AWS infrastructure setup (App Runner, RDS/PostgreSQL, S3, Bedrock/Textract, SQS, Lambda).
    *   CI/CD pipeline basics.
    *   **Database schema design & initial implementation:** 
        *   Identify core entities (Users, Organizations, Invoices, POs, Approval Status, Audit Logs).
        *   Define relationships and initial field requirements for MVP features.
        *   Implement foundational tables in PostgreSQL (AWS RDS).
    *   User authentication setup.
    *   Basic UI shell (Next.js) & PDF upload component.
    *   Integrate AWS Textract (OCR) & LLM (AWS Bedrock/Claude or OpenAI API) for initial data extraction.
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
    *   Develop API integration POC/implementation for **one** target accounting system (e.g., QuickBooks or Sage Intacct).
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
*   The implemented simple PO data management mechanism meets MVP needs.
*   Approval logic requirements are relatively simple for the MVP.
*   Basic error flagging is sufficient for initial MVP operational needs.
*   AWS Bedrock/Claude and OpenAI API both provide sufficient accuracy for invoice data extraction. 