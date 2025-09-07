# Career Mentor

## Project Overview

Career Mentor is an intelligent web application designed to help users discover suitable career paths by analyzing the skills extracted from their resumes. The application combines machine learning predictions with rule-based recommendations to provide personalized career guidance. It also offers detailed career roadmaps and insights into current industry trends to support informed decision-making and skill development.

---

## Problem Statement

Navigating the increasingly complex job market is a challenge for many individuals, as they struggle to identify career opportunities aligned with their skills and interests. Additionally, access to personalized and actionable career guidance is often limited or costly.

Traditional career counseling services and generic online advice fail to provide tailored recommendations based on an individual’s unique resume and skill set. Users also lack clear pathways and up-to-date information about skills and trends required to thrive in their chosen fields.

Career Mentor addresses these challenges by:

- **Extracting skills** from resumes in multiple formats (TXT, PDF, DOCX) through robust text processing.
- **Predicting relevant career paths** using a trained machine learning model.
- **Providing rule-based career recommendations** to complement model predictions.
- **Delivering step-by-step career roadmaps** that clearly outline the skills and technologies to learn.
- **Tracking user progress** with interactive roadmap checklists.
- **Sharing current and emerging industry trends** to keep users informed and competitive.

---

## Key Features

- Multi-format resume upload and text extraction
- Hybrid skill identification (ML-based and rule-based)
- Top career prediction with confidence scores
- Alternative career suggestions based on skills overlap
- Interactive, detailed career roadmaps with progress tracking
- Industry trends dashboard for real-time insights
- Clean, responsive user interface with modern glassmorphic design

---

## Technology Stack

- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn model pipeline for career prediction
- **File Processing:** PyPDF2 (PDF), python-docx (DOCX), native file reading (TXT)
- **Frontend:** HTML, CSS, Jinja2 templating
- **Session Management:** Flask sessions for storing user progress

---

## Installation and Usage

1. Clone the repository.
2. Create a Python virtual environment and activate it.
3. Install dependencies:
