**OVERVIEW**


The Quizify project is engineered to harness the capabilities of AI tools such as the Gemini Pro and VertexAI embedding models to develop a robust quiz generation and grading method. 
This method seamlessly accepts a PDF file, a designated topic, and the desired number of questions as inputs. 
Upon ingestion, the document undergoes processing by a specialized document processor, followed by embedding, chunking, 
and storage of the resultant chunks in ChromaDB as a vector store. Utilizing this vector store alongside the specified topic and question count, 
the Gemini Pro efficiently generates tailored questions.

The user interface (UI), built using Streamlit, offers a user-friendly experience. 
It comprises a sleek form where users input the document, topic, and desired number of questions. Upon submission, the questions elegantly unfold in a linear fashion, 
accompanied by smooth transitions, as demonstrated in the provided video.
To kick off the process:

Begin by cloning the repository.
Once cloned, navigate to the repository directory in your terminal. Then, execute the following command to swiftly install the required libraries and dependencies:

```python
pip3 install -r requirements.txt
```

Obtain the JSON key file and configure Vertex AI. Here's a step-by-step guide:

3.1. Log into the Google Cloud Console.

3.2. Create a new project.

3.3. Enable the Vertex AI API.

3.4. Create a service account.

3.5. Add the generated key.

3.6. Assign the necessary permissions, such as owner, service account token creator, and Vertex AI administrator.

Utilize the downloaded key and integrate its path into your program.
Finally, execute the command:
```python
streamlit run task_10.py
```
This sequence will set you on the right track.
