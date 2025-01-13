import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_keywords(text):
    """Extract significant words/phrases as keywords."""
    return text.lower().replace("\n", " ").split()

def calculate_similarity(resume, job_description):
    """Calculate similarity percentage between resume and job description."""
    vectorizer = CountVectorizer().fit_transform([resume, job_description])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return round(cosine_sim * 100, 2)

def find_missing_keywords(resume_keywords, job_keywords):
    """Identify keywords missing from the resume."""
    return list(set(job_keywords) - set(resume_keywords))

def main():
    st.title("Resume Matching ATS")
    st.write("Evaluate a resume against a job description.")

    with st.sidebar:
        st.header("Upload or Paste Documents")
        resume_input = st.text_area("Paste Resume Text Here", height=300)
        job_desc_input = st.text_area("Paste Job Description Here", height=300)

    if resume_input and job_desc_input:
        # Preprocessing
        resume_keywords = extract_keywords(resume_input)
        job_keywords = extract_keywords(job_desc_input)

        # Calculate similarity percentage
        match_percentage = calculate_similarity(resume_input, job_desc_input)

        # Find missing keywords
        missing_keywords = find_missing_keywords(resume_keywords, job_keywords)

        # Display results
        st.header("Results")
        st.metric(label="Match Percentage", value=f"{match_percentage}%")

        st.subheader("Missing Keywords")
        if missing_keywords:
            st.write(", ".join(missing_keywords))
        else:
            st.write("No significant keywords are missing!")

        st.subheader("Final Thoughts")
        if match_percentage > 80:
            st.success("The resume aligns very well with the job description. Consider this candidate for the role.")
        elif 50 < match_percentage <= 80:
            st.warning("The resume matches moderately. Review if the missing keywords are critical for the role.")
        else:
            st.error("The resume has a low alignment with the job description. The candidate may not be suitable.")

if __name__ == "__main__":
    main()
