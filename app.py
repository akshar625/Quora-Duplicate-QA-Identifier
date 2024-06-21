import streamlit as st
import helper
import pickle

# Load the CountVectorizer and the trained model
cv = pickle.load(open('cv.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def main():
    st.title("Quora Duplicate Questions Detection")
    st.write("")

    q1 = st.text_input("Enter the first question:")
    q2 = st.text_input("Enter the second question:")

    if st.button("Predict"):
        if q1.strip() == "" or q2.strip() == "":
            st.error("Please enter both questions.")
        else:
            # Call the query_point_creator function with CountVectorizer as argument
            query_point = helper.query_point_creator(q1, q2, cv)
            prediction = model.predict(query_point)
            
            st.write("")
            st.write("---")
            st.write("")
            st.write("### Prediction:")
            st.write("")

            if prediction:
                st.success("Duplicate")
            else:
                st.info("Not Duplicate")

if __name__ == "__main__":
    main()
