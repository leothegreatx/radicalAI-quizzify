**OVERVIEW**
The Quizify project is engineered to harness the capabilities of AI tools such as the Gemini Pro and VertexAI embedding models to develop a robust quiz generation method. 
This method seamlessly accepts a PDF file, a designated topic, and the desired number of questions as inputs. 
Upon ingestion, the document undergoes processing by a specialized document processor, followed by embedding, chunking, 
and storage of the resultant chunks in ChromaDB as a vector store. Utilizing this vector store alongside the specified topic and question count, 
the Gemini Pro efficiently generates tailored questions.

The user interface (UI), built using Streamlit, offers a user-friendly experience. 
It comprises a sleek form where users input the document, topic, and desired number of questions. Upon submission, the questions elegantly unfold in a linear fashion, 
accompanied by smooth transitions, as demonstrated in the provided video.
