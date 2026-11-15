# Streamlit City Assistant

Python 3.10+. Your Groq agent with weather, recent news, chat history, and tool approval.

## Run locally

1. Extract this folder and open a terminal inside it.
2. Create a virtual environment: `python -m venv .venv`
3. Activate it: Windows PowerShell `.venv\Scripts\Activate.ps1`; macOS/Linux `source .venv/bin/activate`.
4. Install: `python -m pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and enter your three API keys.
6. Launch: `python -m streamlit run app.py`

Example: “Give me the weather and latest news in Bokaro.” Review the displayed tool name and arguments, choose Approve or Deny for each request, and submit. A follow-up such as “What about Ranchi?” retains conversation context.

## What changed

- Replaced terminal input approval with LangChain human-in-the-loop middleware, a checkpointer, and Streamlit forms. Tool execution pauses until decisions are submitted.
- Stored the agent and checkpoint thread in session state so Streamlit reruns retain pending approvals and chat history.
- Added HTTPS, a weather request timeout, service error messages, and API key status.
- Removed unused Mistral and Rich imports.
- News uses Tavily's news topic and past-week filter; search coverage and publication dates vary.

Both tools require approval. Missing tool keys produce a clear tool error. API requests use your own service quotas. The Groq model must support tool calling and be available to your account. Keys can alternatively go in Streamlit secrets for deployment; never commit them. Environment values take precedence.

History is in memory, isolated per Streamlit session; it is not durable across server restarts. On an agent failure use New chat; the app does not automatically replay a failed operation. New chat discards pending approvals.

Validation: Python source compilation only; live Groq, Tavily, and OpenWeather calls require your credentials. Dependency ranges are not a tested lockfile.
