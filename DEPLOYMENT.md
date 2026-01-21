# How to Host Your RAG Bot on Streamlit Cloud

The easiest way to host this app for free is using **Streamlit Community Cloud**.

## 1. Prepare Your Code
Before hosting, we must ensure your secret API keys are safe and your dependencies are ready.

1.  **Ignore Sensitive Files**: Ensure you have a `.gitignore` file that excludes `.env`, `venv/`, and your data folders.
2.  **Push to GitHub**: Upload this project to a GitHub repository.

## 2. Deploy on Streamlit Cloud
1.  Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2.  Click **"New app"**.
3.  Select your GitHub repository, branch (usually `main`), and the main file path: `src/app.py`.
4.  Click **"Deploy!"**.

## 3. Configure Secrets (CRITICAL)
Your app will fail immediately because it doesn't have your OpenAI API Key. You must set this in the cloud dashboard.

1.  On your deployed app dashboard, go to the **Settings** menu (three dots similar to the top right).
2.  Select **"Secrets"**.
3.  Paste the contents of your local `.env` file, but format it like TOML:
    ```toml
    OPENAI_API_KEY = "sk-..."
    KMP_DUPLICATE_LIB_OK = "TRUE"
    ```
4.  Save. The app will restart and should now work!

## 4. Note on Persistence
Streamlit Cloud is *ephemeral*.
-   This means if the app goes to sleep, **your uploaded files and the vector index will disappear**.
-   You will need to re-upload and re-process your PDFs each time you visit the app after a break.
-   **Solution**: For a permanent app, you would need to host the Vector Database externally (e.g., Pinecone or Weaviate) instead of saving it to the local `faiss_index` folder.
