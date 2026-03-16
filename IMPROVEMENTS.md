# Future Improvements
If given more time, I would improve the project in the following ways:

1. **Vector Database Integration**  
   Currently, embeddings are stored in memory. I would integrate a vector database
   such as FAISS or Chroma to improve scalability and performance for large documents.

2. **Better Text Chunking**  
   The current chunking is character-based. I would improve this by using
   sentence- or paragraph-based chunking to preserve semantic meaning.

3. **OCR Support for Scanned PDFs**  
   Some PDFs contain scanned images instead of text. I would add OCR support
   using tools like Tesseract so the app can handle image-based PDFs.

4. **Multiple Document Support**  
   I would allow users to upload and query multiple documents at once instead
   of only a single file.

5. **Answer Citations**  
   I would display which section or chunk of the document was used to generate
   each answer, improving transparency.

6. **Model Selection and Streaming Responses**  
   I would add an option to switch between different LLMs and enable streaming
   responses for better user experience.

7. **UI Enhancements**  
   Improvements such as chat history export, better error messages, and
   document previews could further enhance usability.
