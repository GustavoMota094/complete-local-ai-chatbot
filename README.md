

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/complete-local-ai-chatbot.git
    cd complete-local-ai-chatbot
    ```

2.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

3.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

4.  **Install PyTorch separately (IMPORTANT):**
    This project requires a specific build of PyTorch. Install it using the following command (this example is for CUDA 12.8, adjust if your environment differs):
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ```
    *(Note: If you don't have a GPU or compatible CUDA, you might need a CPU-only version. Refer to the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) to find the correct command for your system.)*

5.  **Install other backend dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

6.  **Set up environment variables:**
    Copy `backend/.env.example` to `backend/.env` and fill in your configuration values.
    ```bash
    cp .env.example .env
    # Now edit .env with your details
    ```