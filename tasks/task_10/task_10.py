import streamlit as st
import os
import sys
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient
from tasks.task_5.task_5 import ChromaCollectionCreator
from tasks.task_8.task_8 import QuizGenerator
from tasks.task_9.task_9 import QuizManager

if __name__ == "__main__":
    
    embed_config = { "model_name": "textembedding-gecko@003", "project": "radical-ai", "location": "us-central1", "key_file_path": "/Users/adigweleo/Downloads/radical-ai-678b5d1ae214.json" }
    
    # Add Session State
    if 'question_bank' not in st.session_state or len(st.session_state['question_bank']) == 0:
        
        # Initialize the question bank list in st.session_state
        st.session_state["question_bank"] = []
    
        screen = st.empty()
        with screen.container():
            st.header("Quiz Builder")
            
            # Create a new st.form flow control for Data Ingestion
            with st.form("Load Data to Chroma"):
                st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
                
                processor = DocumentProcessor()
                processor.ingest_documents()
            
                embed_client = EmbeddingClient(**embed_config)
            
                chroma_creator = ChromaCollectionCreator(processor, embed_client)
                
                # Set topic input and number of questions
                topic_input = st.text_input("Enter the topic:")
                num_questions = st.number_input("Number of questions:", min_value=1, max_value=10)
                    
                submitted = st.form_submit_button("Submit")
                
                if submitted:
                    chroma_creator.create_chroma_collection()
                        
                    if len(processor.pages) > 0:
                        st.write(f"Generating {num_questions} questions for topic: {topic_input}")
                    
                    # Initialize a QuizGenerator class
                    quiz_generator = QuizGenerator(topic_input, num_questions, chroma_creator)
                    # Generate quiz questions
                    st.session_state["question_bank"] = quiz_generator.generate_quiz()
                    # Initialize question index
                    st.session_state["question_index"] = 0

    elif "display_quiz" not in st.session_state or st.session_state["display_quiz"]:
        
        st.empty()
        with st.container():
            st.header("Generated Quiz Question: ")
            quiz_manager = QuizManager(st.session_state["question_bank"])
            
            # Format the question and display it
            with st.form("MCQ"):
                # Get the current question index
                index_question = quiz_manager.get_question_at_index(st.session_state["question_index"])
                
                # Unpack choices for radio button
                choices = [f"{choice['key']}) {choice['value']}" for choice in index_question['choices']]
                
                # Display the Question
                st.write(f"{st.session_state['question_index'] + 1}. {index_question['question']}")
                answer = st.radio(
                    "Choose an answer",
                    choices,
                    index=None
                )
                
                # Check if an answer was submitted
                answer_choice = st.form_submit_button("Submit")
                
                # Display feedback if an answer was submitted
                if answer_choice and answer is not None:
                    correct_answer_key = index_question['answer']
                    if answer.startswith(correct_answer_key):
                        st.success("Correct!")
                    else:
                        st.error("Incorrect!")
                    st.write(f"Explanation: {index_question['explanation']}")
                
                # Navigate to the next question
                if st.form_submit_button("Next Question"):
                    st.session_state["question_index"] += 1
                if st.form_submit_button("Previous Question"):
                    st.session_state["question_index"] -= 1
                
                

                # Reset the question index when all questions have been answered
                if st.session_state["question_index"] >= len(st.session_state["question_bank"]):
                    st.session_state["display_quiz"] = False
                    st.session_state["question_index"] = 0

