# Deployment

## Vercel: FastAPI API

Vercel can host the FastAPI API as a Python serverless function. It cannot run the Streamlit dashboard as a persistent process.

1. Push this project to GitHub.
2. Import the repository at <https://vercel.com/new>.
3. Keep the repository root as the project root.
4. Leave the framework preset as **Other**.
5. Deploy. Vercel uses `vercel.json` and `api/index.py` automatically.
6. Test the generated URL:

```text
https://YOUR-APP.vercel.app/health
https://YOUR-APP.vercel.app/docs
```

The API is stateless on Vercel. The local SQLite database is not durable between deployments or serverless instances. Use a hosted PostgreSQL database or another external storage service before relying on saved documents in production.

## Streamlit dashboard

Deploy the dashboard separately to Streamlit Community Cloud:

1. Open <https://share.streamlit.io/>.
2. Select the GitHub repository and branch.
3. Set the main file to `dashboard/app.py`.
4. Deploy.

For a production dashboard, configure an API base URL and call the deployed Vercel API instead of using local SQLite. The current dashboard is designed for local use and Streamlit deployment with its own local database.

## Local verification before deployment

```powershell
python -m pytest -q
python -m compileall -q src api dashboard evaluation tests
```
