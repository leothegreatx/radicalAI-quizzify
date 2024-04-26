import streamlit as st
from langchain.document_loaders import PyPDFLoader
import os
import tempfile
from io import BytesIO

class DocumentProcessor:
    """This class encapsulates the functionality for processing uploaded PDF documents
    using Streamlit and Langchain's PyPDFLoader. It provides a method to render a
    file uploader widget, process the uploaded PDF files, extract their pages,
    and display the total number of pages extracted.
    """

    def __init__(self):
        self.pages = []  # List to keep track of pages from all documents

    def ingest_documents(self):
        """
        Renders a file uploader in a Streamlit app, processes uploaded PDF files,
        extracts their pages, and updates the self.pages list with the total
        number of pages.
        """
        # Step 1: Render a file uploader widget.
        uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True, type="pdf")

        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                file_data = BytesIO(uploaded_file.read())
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file_data.getbuffer())
                    temp_file_path = temp_file.name

                # Initialize PyPDFLoader with the temporary file path
                pdf_loader = PyPDFLoader(temp_file_path)

                # Process the temporary file using PyPDFLoader
                extracted_pages = pdf_loader.load()

                # Add the extracted pages to the pages class variable
                self.pages.extend(extracted_pages)

                # Clean up by deleting the temporary file
                os.unlink(temp_file_path)

            # Display the total number of pages processed
            st.write(f"Total pages processed: {len(self.pages)}")

            return self.pages

    def get_text(self, page_number):
        """
        Retrieve the text content of a specific page from the processed documents.
        :param page_number: The page number for which to retrieve the text content.
        :return: The text content of the specified page.
        """
        if page_number < len(self.pages):
            # Assuming each page is represented as a string in the self.pages list
            return str(self.pages[page_number])
        else:
            return "Page number out of range"

if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.ingest_documents()

