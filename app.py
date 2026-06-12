import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎓 College FAQ Chatbot")
st.write("Ask any college-related question.")

faq_questions = [
    "What are the college timings?",
    "Where is the library located?",
    "How can I pay my fees?",
    "What courses are offered?",
    "How do I apply for a scholarship?",
    "What is the attendance requirement?",
    "How can I contact the administration?",
    "When are the semester exams conducted?",
    "How do I access the student portal?",
    "What are the hostel facilities?",
    "Is transportation available?",
    "How can I get my hall ticket?",
    "How do I check my results?",
    "What documents are required for admission?",
    "How can I apply for internships?",
    "What clubs are available in the college?",
    "How do I register for events?",
    "Where is the computer lab?",
    "How can I reset my student portal password?",
    "What is the procedure for leave application?"
]

faq_answers = [
    "College timings are from 9:00 AM to 4:00 PM.",
    "The library is located near the academic block.",
    "Fees can be paid through the online student portal or at the accounts office.",
    "The college offers Engineering, Management, and Science courses.",
    "Scholarship applications can be submitted through the scholarship cell.",
    "Students must maintain at least 75% attendance.",
    "You can contact the administration through the college office.",
    "Semester exams are usually conducted at the end of each semester.",
    "The student portal can be accessed using your student credentials.",
    "Hostel facilities include accommodation, Wi-Fi, and dining services.",
    "Yes, transportation is available on selected routes.",
    "Hall tickets can be downloaded from the student portal.",
    "Results are published on the college website and student portal.",
    "Admission requires academic certificates, ID proof, and photographs.",
    "Students can apply for internships through the placement cell.",
    "Various technical, cultural, and sports clubs are available.",
    "Event registration can be completed through the student portal.",
    "The computer lab is located in the technology block.",
    "Use the 'Forgot Password' option on the student portal.",
    "Leave applications should be submitted to the class advisor."
]

user_question = st.text_input("Ask your question:")

if st.button("Get Answer"):

    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(
            faq_questions + [user_question]
        )

        similarity = cosine_similarity(
            vectors[-1],
            vectors[:-1]
        )

        best_match = similarity.argmax()
        score = similarity[0][best_match]

        if score > 0.3:
            st.success("Answer:")
            st.write(faq_answers[best_match])
        else:
            st.warning(
                "Sorry, I don't know the answer to that question."
            )