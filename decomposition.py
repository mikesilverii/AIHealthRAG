def decompose_query(main_query: str, llm):
    """Use LLM to generate subqueries from a general medical query."""
    prompt = (
        f"Decompose the following query into a list of 3 to 6 specific subquestions, "
        f"each focusing on a different aspect of a patient’s medical history.\n\n"
        f"Query: {main_query}\n\n"
        f"Respond with a numbered list of subquestions."
    )
    response = llm.complete(prompt)
    subqueries = response.text.strip().split("\n")
    subqueries = [q.split(".", 1)[-1].strip() for q in subqueries if "." in q]
    return subqueries

#use output as an example

all_subqueries = [
    # Demographics
    "What is the patient’s full name?",
    "What is the patient’s date of birth?",
    "What is the patient’s medical record number (MRN)?",
    "What is the patient’s gender?",
    "Who is the patient’s Primary Care Provider, Primary Care Doctor (PCP)? Be sure that this is obviously mentioned", #primary care provider
    "What is the patient’s code status?",
    "What is the patient’s contact information?",
    "Who are the patient’s parents or guardians?",
    "What is the patient’s email address?",

    # Clinical History
    #"What is the patient’s significant past medical history? Must be Y or Yes or True. Use bullet points when possible. Be clear and concise.",
    "What are the patient's medical conditions?",
    "What are the patient’s past hospitalizations? Indicate which are surgical admissions. Use bullet points when possible. Be clear and concise.",
    "What surgeries has the patient undergone, with year? Format as: Date: Surgery: Location/Hosptial. Use bullet points when possible. Be clear and concise.", #where did they take place
    "What reported or reviewed allergies does the patient have? Use bullet points when possible.",
    "What is the patient’s family medical history and social history? Use bullet points when possible. Be clear and concise.", #combine these?
    #"What is the patient’s social, non-family history? Use bullet points when possible.",

    # Specialists & Diagnostics
    "Which specialists has the patient seen? Be as specific as possible. Format as: Name - Role/Relation to Patient - Specialization. Be clear and concise.", #
    "What imaging studies has the patient undergone? Use bullet points when possible.",
    "What diagnostic studies, tests, or recent blood/lab tests has the patient had? Show the actual recent blood test result values. Use bullet points when possible.",

    # Follow-Up and Future Guidance
    #"What are the general care guidelines for a patient with this condition? Nothing post-operation related. Be clear and concise. Use bullet points when possible."
    "Based on this patient's medical history, what are the general future care guidelines *unrelated* to any past surgical procedures or hospitalizations? Focus on the ongoing management of their chronic conditions. Be clear and concise. Use bullet points if applicable."
    #"What are the future follow-up recommendations for the patient? We don't need things from the past. Use bullet points when possible.",
    #"What are the general, future care guidelines for this patient? DO NOT mention things based on time or based on a surgery. Only use very general guidelines for the patient. Use bullet points when possible.",
]

section_titles = [
    "Patient Name",
    "Date of Birth",
    "MRN",
    "Gender",
    "PCP",
    "Code Status",
    "Contact",
    "Parents/Guardians",
    "Email",
    "Past Medical History",
    "Past Hospitalizations",
    "Past Surgical History",
    "Allergies",
    "Family and Social History",
    #"Social History",
    "Specialists",
    "Imaging",
    "Diagnostic Studies",
    #"Diagnostic Studies",
    #"Follow-up Recommendations / Summary",
    "Future Care Guidelines",
]

