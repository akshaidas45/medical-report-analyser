import streamlit as st
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
import tempfile
import re
import os
from fpdf import FPDF

# Load summarization model using LangChain
llm = HuggingFacePipeline.from_model_id(model_id="facebook/bart-large-cnn", task="summarization")

prompt_template = PromptTemplate.from_template(
    "Summarize the following medical text:\n\n{text}"
)

def summarize_text(text):
    chain = prompt_template | llm
    return chain.invoke({"text": text})

# Medicines for common conditions
BEST_MEDICINES = {
    "anemia": ["Ferrous sulfate", "Folic acid"],
    "polycythemia": ["Phlebotomy"],
    "leukopenia": ["Immunostimulants"],
    "infection": ["Amoxicillin", "Azithromycin"],
    "thrombocytopenia": ["Platelet transfusion"],
    "thrombocytosis": ["Hydroxyurea"],
    "hypoglycemia": ["Glucose tablets"],
    "diabetes": ["Metformin", "Insulin"],
    "kidney dysfunction": ["Consult nephrologist"],
    "cholesterol": ["Rosuvastatin", "Atorvastatin"],
    "liver inflammation": ["Liver protective agents"],
    "proteinuria": ["Treat underlying cause"],
    "fever": ["Paracetamol", "Ibuprofen"],
    "common cold": ["Cetirizine", "Levocetirizine"],
    "cough": ["Dextromethorphan", "Bromhexine"],
    "headache": ["Paracetamol", "Aspirin"],
    "body pain": ["Ibuprofen", "Diclofenac"],
    "sore throat": ["Salt water gargle", "Azithromycin"],
    "constipation": ["Lactulose", "Isabgol"],
    "diarrhea": ["ORS", "Loperamide"],
    "acidity": ["Omeprazole", "Pantoprazole"],
    "nausea": ["Ondansetron", "Domperidone"],
    "vomiting": ["Ondansetron", "Domperidone"],
    "flu": ["Oseltamivir", "Paracetamol"],
    "sinusitis": ["Amoxicillin", "Nasal corticosteroids"],
    "hypertension": ["Amlodipine", "Losartan"],
    "hypotension": ["Fludrocortisone", "Midodrine"],
    "dehydration": ["ORS", "IV fluids"],
    "allergies": ["Antihistamines", "Corticosteroids"],
    "bronchitis": ["Bronchodilators", "Steroids"],
    "asthma": ["Salbutamol", "Inhaled corticosteroids"],
    "pneumonia": ["Levofloxacin", "Ceftriaxone"],
    "urinary tract infection": ["Nitrofurantoin", "Ciprofloxacin"],
    "gout": ["Colchicine", "Allopurinol"],
    "arthritis": ["NSAIDs", "Methotrexate"],
    "migraine": ["Sumatriptan", "Rizatriptan"],
    "osteoporosis": ["Bisphosphonates", "Calcium supplements"],
    "depression": ["SSRIs", "SNRIs"],
    "anxiety": ["Benzodiazepines", "Buspirone"],
    "insomnia": ["Melatonin", "Zolpidem"]
}

# Normal ranges for test results
NORMAL_RANGES = {
    "hemoglobin": (13.5, 17.5),
    "rbc": (4.7, 6.1),
    "wbc": (4000, 11000),
    "platelets": (150000, 450000),
    "hematocrit": (40, 50),

    "glucose_fasting": (70, 100),
    "glucose_postprandial": (70, 140),
    "hba1c": (4.0, 5.6),  

    "cholesterol_total": (0, 200),
    "hdl": (40, 60),
    "ldl": (0, 130),
    "triglycerides": (0, 150),

    "bun": (7, 20),
    "creatinine": (0.6, 1.3),

    "sodium": (135, 145),
    "potassium": (3.5, 5.1),
    "calcium": (8.6, 10.2),
    "iron": (60, 170),

    "bilirubin": (0.1, 1.2),
    "alkaline_phosphatase": (44, 147),
    "alt": (7, 56),
    "ast": (10, 40),

    "tsh": (0.4, 4.0),
    "vitamin_d": (20, 50),

    "blood_pressure": {
        "systolic": (90, 120),
        "diastolic": (60, 80)
    }
}

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    loader = PyMuPDFLoader(file_path=tmp_path)
    docs = loader.load()
    full_text = "\n".join(doc.page_content for doc in docs)
    os.remove(tmp_path)
    return full_text

# Extract patient details
def extract_patient_details(text):
    name_match = re.search(r"Name[:\s]*([A-Za-z]+(?: [A-Za-z]+)?)\s", text)
    age_match = re.search(r"Age[:\s]*(\d+)", text)

    name = name_match.group(1).strip() if name_match else "Unknown"
    age = age_match.group(1).strip() if age_match else "Unknown"

    return f"Name: {name}\nAge: {age}"

# Detect conditions in the text
def detect_conditions(text):
    return [cond for cond in BEST_MEDICINES if cond.lower() in text.lower()]

# Extract abnormal test results
def extract_test_results(text):
    abnormal = {}
    for test, (low, high) in NORMAL_RANGES.items():
        match = re.search(rf"{test}[:\s]*([\d.]+)", text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if value < low or value > high:
                abnormal[test] = value
    return abnormal

# Create a PDF report including test results
def create_pdf(patient_info, summary, recommendations, test_results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.multi_cell(0, 10, "--- Patient Details ---")
    pdf.multi_cell(0, 10, patient_info)
    pdf.ln()

    pdf.multi_cell(0, 10, "--- Summary ---")
    pdf.multi_cell(0, 10, summary)
    pdf.ln()

    pdf.multi_cell(0, 10, "--- Recommended Medicines ---")
    if recommendations:
        for cond, meds in recommendations.items():
            pdf.multi_cell(0, 10, f"Condition: {cond.capitalize()}")
            pdf.multi_cell(0, 10, f"Medicines: {', '.join(meds)}")
            pdf.ln()
    else:
        pdf.multi_cell(0, 10, "No specific medicines found.")
    
    pdf.ln()

    pdf.multi_cell(0, 10, "--- Test Results ---")
    if test_results:
        for test, value in test_results.items():
            pdf.multi_cell(0, 10, f"{test.capitalize()}: {value} (Out of normal range)")
            pdf.ln()
    else:
        pdf.multi_cell(0, 10, "All test results appear within normal ranges.")
    
    file_path = "final_report.pdf"
    pdf.output(file_path)
    return file_path

# Streamlit app UI
st.set_page_config(page_title="🩺 Medical Report Analyzer", layout="centered")

st.markdown("""
    <h2 style='text-align: center;'>🩺 Medical Report Analyzer</h2>
    <p style='text-align: center;'>Upload your medical report (PDF) to analyze and get medicine recommendations.</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📎 Upload Medical Report (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("🔍 Reading report..."):
        report_text = extract_text_from_pdf(uploaded_file)
        patient_details = extract_patient_details(report_text)

    st.markdown("### 📋 Patient Details")
    st.code(patient_details, language='text')

    if st.button("🧪 Analyze Report"):
        with st.spinner("✍️ Summarizing medical content..."):
            summary_output = summarize_text(report_text[:4000])

        st.markdown("### 📝 Summary")
        st.success(summary_output)

        with st.spinner("🔎 Detecting conditions & medicines..."):
            conditions = detect_conditions(report_text)
            recommendations = {cond: BEST_MEDICINES[cond] for cond in conditions}

        st.markdown("### 💊 Recommended Medicines")
        if recommendations:
            for cond, meds in recommendations.items():
                st.markdown(f"🔹 **Condition:** `{cond.capitalize()}`")
                st.markdown(f"✅ **Medicines:** {', '.join(meds)}")
        else:
            st.info("🤔 No matching condition detected for medicine suggestions.")

        st.markdown("### 📈 Test Results Analysis")
        abnormal_results = extract_test_results(report_text)
        if abnormal_results:
            for test, value in abnormal_results.items():
                st.markdown(f"⚠️ **{test.capitalize()}**: `{value}` (Out of normal range)")
        else:
            st.info("✅ All test results appear within normal ranges.")

        pdf_file = create_pdf(patient_details, summary_output, recommendations, abnormal_results)
        with open(pdf_file, "rb") as f:
            st.download_button(
                label="📥 Download Full Report (PDF)",
                data=f,
                file_name=os.path.basename(pdf_file),
                mime="application/pdf"
            )
